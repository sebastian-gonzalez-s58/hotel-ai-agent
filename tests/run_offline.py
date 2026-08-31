"""Run unit regressions with outbound sockets blocked, including live model evaluations."""
import inspect
import os
import socket
import unittest
from unittest.mock import patch


def main():
    original_connect = socket.socket.connect

    def blocked_connect(sock, address):
        # Windows implements the event loop's internal socketpair over loopback TCP.
        caller = inspect.currentframe().f_back
        socketpair = getattr(socket, "_fallback_socketpair", None)
        if socketpair and caller.f_code is socketpair.__code__:
            return original_connect(sock, address)
        raise AssertionError("Live network disabled by offline test runner")

    with patch.dict(os.environ, {"RUN_LIVE_SCOPE_EVALS": "0"}), \
            patch.object(socket.socket, "connect", blocked_connect), \
            patch.object(socket.socket, "connect_ex", blocked_connect):
        suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
