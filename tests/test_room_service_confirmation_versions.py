"""BC-006: old, used and unversioned menus cannot authorize a current order."""
import json
import unittest
from uuid import uuid4

from app.agents.v2_turn_planner import plan_v2_turn, _validate_plan
from app.core.errors import AgentModelError
from tests.conversation_regression.library import Event, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.replay import replay_case
from tests.conversation_regression.session import ConversationSession, structured_summary
from tests.conversation_regression.room_confirmation_support import current_room_button


class RoomConfirmationVersionsTest(unittest.TestCase):
    def setUp(self):
        self.cases = {c.id: c for c in load_cases()}
        self.case = self.cases["SC-009-stale-confirmation"]
        self.profile = profiles()["hotel"]

    def turn(self, session, event, replies=()):
        request = session.receive(event)
        before = request.model_dump(mode="json")
        with scripted_runtime(replies):
            response = plan_v2_turn(request)
        self.assertEqual(before, request.model_dump(mode="json"))
        session.accept(response)
        return request, response

    def presented(self, edited=False):
        session = ConversationSession(self.case, self.profile)
        for step in self.case.steps[:2 if edited else 1]:
            self.turn(session, step.event, step.model_replies)
        return session

    def test_old_current_and_used_buttons_complete_full_conversation(self):
        result = replay_case(self.case, self.profile)
        self.assertEqual("OFFLINE_PARTIAL_PASS", result["status"], result)

    def test_old_confirm_change_cancel_preserve_edited_order(self):
        for action in ("CONFIRM", "CHANGE", "CANCEL"):
            with self.subTest(action=action):
                session = self.presented(edited=True)
                state = structured_summary(session.request.conversation.summary)
                _, result = self.turn(session, Event(kind="guest", text=action,
                    reply_id=current_room_button(session.responses[0], action)))
                after = structured_summary(result.updatedConversationSummary)
                self.assertEqual(state["capturedFields"], after["capturedFields"])
                self.assertEqual([], result.toolCalls)
                self.assertNotEqual(state["roomServiceConfirmation"], after["roomServiceConfirmation"])

    def test_current_change_and_cancel_still_work(self):
        for action in ("CHANGE", "CANCEL"):
            with self.subTest(action=action):
                session = self.presented()
                _, result = self.turn(session, Event(kind="guest", text=action,
                    reply_id=current_room_button(session.responses[-1], action)))
                after = structured_summary(result.updatedConversationSummary)
                self.assertFalse(after.get("awaitingExplicitConfirmation"))
                self.assertNotIn("roomServiceConfirmation", after)
                if action == "CANCEL":
                    self.assertEqual({}, after)
                else:
                    self.assertEqual(1, after["capturedFields"]["items"][0]["quantity"])

    def test_unversioned_migration_requires_new_confirmation(self):
        for old in ("confirmation:ROOM_SERVICE:CONFIRM", "room-service:confirm"):
            with self.subTest(old=old):
                session = self.presented()
                state = structured_summary(session.request.conversation.summary)
                state.pop("roomServiceConfirmation")
                session.request.conversation.summary = json.dumps(state)
                _, result = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=old))
                self.assertEqual([], result.toolCalls)
                _, current = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=current_room_button(result)))
                self.assertEqual(["START_SERVICE"], [c.toolName.value for c in current.toolCalls])

    def test_old_button_after_cancellation_cannot_resurrect_order(self):
        session = self.presented()
        old = current_room_button(session.responses[-1])
        self.turn(session, Event(kind="guest", text="Cancelar", reply_id=current_room_button(session.responses[-1], "CANCEL")))
        _, result = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=old))
        self.assertEqual({}, structured_summary(result.updatedConversationSummary))
        self.assertEqual([], result.toolCalls)

    def test_buttons_after_delivery_cannot_claim_cancellation_or_call_model(self):
        from app.schemas.v2_turns import OperationSnapshot
        for reply in ('confirmation:ROOM_SERVICE:used:CANCEL', 'confirmation:ROOM_SERVICE:CANCEL',
                      'room-service:cancel', 'CANCEL_ORDER', 'CHANGE_ORDER', 'CONFIRM_ORDER'):
            with self.subTest(reply=reply):
                session = self.presented()
                session.request.conversation.summary = '{}'
                session.request.recentOperations = [OperationSnapshot(
                    operationId=uuid4(), offeringCode='ROOM_SERVICE', referenceCode='TEST-DELIVERED',
                    lifecycle='COMPLETED', detailedStatus='DELIVERED', summary='Synthetic delivered order',
                    input={'items': [{'name': 'sopa', 'quantity': 1, 'modifications': []}], 'deliveryLocation': 'ROOM'},
                    availableActions=[], pendingConversationTasks=[], version=3)]
                _, response = self.turn(session, Event(kind='guest', text='Cancelar', reply_id=reply))
                self.assertEqual([], response.toolCalls)
                self.assertEqual('{}', response.updatedConversationSummary)
                self.assertNotIn('fue cancelado', ' '.join(m.text for m in response.messages))
                self.assertIn('ya no est', response.messages[0].text)

    def test_legacy_cancel_cannot_discard_a_new_draft(self):
        session = self.presented(edited=True)
        before = structured_summary(session.request.conversation.summary)['capturedFields']
        _, response = self.turn(session, Event(kind='guest', text='Cancelar', reply_id='CANCEL_ORDER'))
        self.assertEqual([], response.toolCalls)
        self.assertEqual(before, structured_summary(response.updatedConversationSummary)['capturedFields'])
        self.assertNotIn('fue cancelado', ' '.join(m.text for m in response.messages))

    def test_same_order_restored_after_edit_does_not_revive_first_menu(self):
        session = self.presented(edited=True)
        edit = self.case.steps[1].model_copy(deep=True)
        edit.event.text = "Que sea una sopa"
        edit.model_replies[0].payload["relevantText"] = edit.event.text
        edit.model_replies[1].payload["edits"][0].update(quantity=1, quantityEvidence="una", evidence="una sopa")
        self.turn(session, edit.event, edit.model_replies)
        _, result = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=current_room_button(session.responses[0])))
        self.assertEqual([], result.toolCalls)
        self.assertEqual(1, structured_summary(result.updatedConversationSummary)["capturedFields"]["items"][0]["quantity"])

    def test_language_only_preserves_current_button_and_exact_order(self):
        session = self.presented()
        old = current_room_button(session.responses[-1])
        state = structured_summary(session.request.conversation.summary)
        language = self.cases["SC-007-language-switch"].steps[0]
        _, response = self.turn(session, language.event, language.model_replies)
        self.assertEqual(state, structured_summary(response.updatedConversationSummary))
        _, result = self.turn(session, Event(kind="guest", text="Confirm", reply_id=old))
        self.assertEqual(state["capturedFields"], result.toolCalls[0].arguments["input"])

    def test_free_text_approval_remains_available(self):
        session = self.presented(edited=True)
        confirm = self.cases["SC-007-language-switch"].steps[1]
        _, result = self.turn(session, confirm.event, confirm.model_replies)
        self.assertEqual(2, result.toolCalls[0].arguments["input"]["items"][0]["quantity"])

    def test_validator_rejects_old_buttons_even_if_planner_proposes_start(self):
        session = self.presented(edited=True)
        request, response = self.turn(session, Event(kind="guest", text="Confirmar",
            reply_id=current_room_button(session.responses[-1])))
        for old in (current_room_button(session.responses[0]), "confirmation:ROOM_SERVICE:CONFIRM"):
            with self.subTest(old=old):
                tampered = request.model_copy(deep=True)
                tampered.conversation.recentMessages[-1].interactionReplyId = old
                with self.assertRaises(AgentModelError):
                    _validate_plan(tampered, response)

    def test_token_cannot_authorize_another_conversation_or_changed_fields(self):
        for change_conversation in (True, False):
            with self.subTest(change_conversation=change_conversation):
                session = self.presented()
                button = current_room_button(session.responses[-1])
                if change_conversation:
                    session.request.conversation.conversationId = uuid4()
                else:
                    state = structured_summary(session.request.conversation.summary)
                    state["capturedFields"]["items"][0]["quantity"] = 9
                    session.request.conversation.summary = json.dumps(state)
                _, response = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=button))
                self.assertEqual([], response.toolCalls)

    def test_validator_rejects_changed_input_and_non_confirm_actions(self):
        session = self.presented()
        menu = session.responses[-1]
        request, response = self.turn(session, Event(kind="guest", text="Confirmar", reply_id=current_room_button(menu)))
        changed = response.model_copy(deep=True)
        changed.toolCalls[0].arguments["input"]["items"][0]["quantity"] = 8
        with self.assertRaises(AgentModelError):
            _validate_plan(request, changed)
        for action in ("CHANGE", "CANCEL"):
            with self.subTest(action=action):
                invalid = request.model_copy(deep=True)
                invalid.conversation.recentMessages[-1].interactionReplyId = current_room_button(menu, action)
                with self.assertRaises(AgentModelError):
                    _validate_plan(invalid, response)


if __name__ == "__main__":
    unittest.main()
