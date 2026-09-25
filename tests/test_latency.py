import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.core import latency
from app.main import run_agent_step
from app.services.openai_client import call_openai_json_result


class LatencyTest(unittest.TestCase):
    def setUp(self):
        self.enabled = patch.object(settings, 'chat_latency_enabled', True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    def rows(self, captured):
        return [json.loads(row.getMessage().split('CHAT_LATENCY ', 1)[1]) for row in captured.records]

    def test_nested_spans_link_and_do_not_log_exception_contents(self):
        with self.assertLogs('chat.latency', 'INFO') as captured:
            with latency.trace('root-1', 'caller-1'):
                with latency.span('outer'):
                    with self.assertRaises(ValueError):
                        with latency.span('inner'):
                            raise ValueError('private guest message')
                self.assertEqual('caller-1', latency.current().span_id)
        inner, outer = self.rows(captured)
        self.assertEqual(outer['span_id'], inner['parent_span_id'])
        self.assertEqual('root-1', inner['trace_id'])
        self.assertEqual('error', inner['outcome'])
        self.assertNotIn('private guest message', str(captured.output))
        self.assertIsNone(latency.current())

    def test_disabled_preserves_results_and_exceptions_without_emitting(self):
        @latency.timed('test')
        def identity(value):
            if isinstance(value, Exception):
                raise value
            return value
        sentinel = object()
        with patch.object(settings, 'chat_latency_enabled', False), self.assertNoLogs('chat.latency'):
            with latency.trace('disabled'):
                self.assertIs(sentinel, identity(sentinel))
                self.assertEqual({}, latency.headers())
                with self.assertRaisesRegex(ValueError, 'same'):
                    identity(ValueError('same'))

    def test_invalid_inbound_trace_cannot_inject_log_lines(self):
        with latency.trace('private\nphone=555', 'unsafe parent'):
            self.assertNotEqual('private\nphone=555', latency.current().trace_id)
            self.assertIsNone(latency.current().span_id)
            self.assertNotIn('\n', str(latency.headers()))

    @patch('app.services.openai_client.record_model_call')
    @patch('app.services.openai_client.get_openai_client')
    def test_model_timing_does_not_change_payload_or_add_calls(self, client, telemetry):
        client.return_value.responses.create.return_value = SimpleNamespace(
            output_text='{"ok":true}', id='test', usage=None)
        with self.assertLogs('chat.latency', 'INFO') as captured, latency.trace('model-test'):
            result = call_openai_json_result('secret prompt', purpose='TEST')
        self.assertEqual({'ok': True}, result.payload)
        client.return_value.responses.create.assert_called_once()
        self.assertEqual('secret prompt', client.return_value.responses.create.call_args.kwargs['input'])
        self.assertNotIn('secret prompt', str(captured.output))
        rows = self.rows(captured)
        self.assertIn('model.sdk_request', [r['stage'] for r in rows])
        self.assertIn('model.parse_json', [r['stage'] for r in rows])

    def test_concurrent_turns_keep_context_in_threadpool(self):
        async def work(identity):
            with latency.trace(identity), latency.span('turn'):
                return await run_agent_step(lambda: latency.current().trace_id)
        async def run():
            with patch('app.main.agent_request_semaphore', asyncio.Semaphore(2)):
                return await asyncio.gather(work('first'), work('second'))
        with self.assertLogs('chat.latency', 'INFO') as captured:
            self.assertEqual(['first', 'second'], asyncio.run(run()))
        self.assertIsNone(latency.current())
        for row in self.rows(captured):
            self.assertIn(row['trace_id'], {'first', 'second'})
        self.assertEqual(2, sum(r['stage'] == 'agent.threadpool_wait' for r in self.rows(captured)))

    def test_cancelled_semaphore_wait_is_logged_without_releasing_unowned_permit(self):
        async def run():
            semaphore = asyncio.Semaphore(0)
            with patch('app.main.agent_request_semaphore', semaphore), latency.trace('cancelled'):
                task = asyncio.create_task(run_agent_step(lambda: self.fail('must not run')))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(semaphore.locked())
        with self.assertLogs('chat.latency', 'INFO') as captured:
            asyncio.run(run())
        self.assertEqual('cancelled', self.rows(captured)[0]['outcome'])

    def test_logging_failure_does_not_mask_success_or_original_error(self):
        with patch.object(latency.logger, 'info', side_effect=RuntimeError('logger failed')):
            with latency.trace('logging'):
                with latency.span('success'):
                    pass
                with self.assertRaisesRegex(ValueError, 'original'):
                    with latency.span('failure'):
                        raise ValueError('original')
        self.assertIsNone(latency.current())
