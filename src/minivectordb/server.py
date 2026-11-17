"""JSON Lines front end: one request per line on stdin, one response per line.

The protocol is described in STDIO_PROTOCOL.md of the task package:

* ``{"type":"ready"}`` is printed once, after recovery, before reading stdin;
* every request gets exactly one response carrying the same ``id`` and a
  boolean ``ok``;
* diagnostics go to stderr, never to stdout;
* ``shutdown`` rolls back the open transaction, persists, then exits with 0.
"""

import json
import sys

from . import errors
from .engine import Engine


def _dumps(message):
    return json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Server:
    def __init__(self, data_dir, stdin=None, stdout=None, stderr=None):
        self.stdin = stdin if stdin is not None else sys.stdin.buffer
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self.engine = None
        self.data_dir = data_dir

    # ------------------------------------------------------------------ output
    def send(self, message):
        self.stdout.write(_dumps(message) + "\n")
        self.stdout.flush()

    def log(self, text):
        self.stderr.write(text.rstrip() + "\n")
        self.stderr.flush()

    # ------------------------------------------------------------------- main
    def serve(self):
        self.engine = Engine(self.data_dir)
        self.send({"type": "ready"})
        while True:
            raw = self.stdin.readline()
            if not raw:
                self.log("stdin closed, shutting down")
                self._shutdown()
                return 0
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except ValueError as exc:
                self.log("invalid request: %s" % exc)
                continue
            if not isinstance(request, dict):
                self.log("request is not an object")
                continue
            if self._handle(request):
                return 0

    def _handle(self, request):
        ident = request.get("id")
        operation = request.get("op", "execute")
        if operation == "shutdown":
            self._shutdown()
            self.send({"id": ident, "ok": True})
            return True
        if operation != "execute":
            self.send(
                {
                    "id": ident,
                    "ok": False,
                    "error": {"code": "unsupported_op", "message": "unknown op %r" % operation},
                }
            )
            return False
        sql = request.get("sql")
        mode = request.get("search_mode", "exact")
        if not isinstance(sql, str):
            self.send(
                {
                    "id": ident,
                    "ok": False,
                    "error": {"code": "syntax_error", "message": "sql must be a string"},
                }
            )
            return False
        try:
            result = self.engine.execute(sql, mode)
        except errors.DBError as exc:
            self.send(
                {"id": ident, "ok": False, "error": {"code": exc.code, "message": exc.message}}
            )
            return False
        except RecursionError:
            self.send(
                {
                    "id": ident,
                    "ok": False,
                    "error": {"code": "syntax_error", "message": "statement is too complex"},
                }
            )
            return False
        except Exception as exc:  # keep the session alive, report the failure
            self.log("internal error: %r" % (exc,))
            self.send(
                {
                    "id": ident,
                    "ok": False,
                    "error": {"code": "internal_error", "message": str(exc)},
                }
            )
            return False
        response = {"id": ident, "ok": True}
        if "columns" in result:
            response["columns"] = result["columns"]
            response["rows"] = result["rows"]
        if "affected_rows" in result:
            response["affected_rows"] = result["affected_rows"]
        self.send(response)
        return False

    def _shutdown(self):
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as exc:  # pragma: no cover - best effort on exit
                self.log("shutdown failed: %r" % (exc,))
            self.engine = None


def run(data_dir):
    return Server(data_dir).serve()
