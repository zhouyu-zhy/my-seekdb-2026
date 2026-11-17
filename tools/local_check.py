#!/usr/bin/env python3
"""Run the official protocol_check driver against this source tree.

The judge talks to a Docker image; during development it is much faster to let
``protocol_check/run.py`` spawn the interpreter directly.  This wrapper only
builds the ``--command`` argument, so shell quoting cannot corrupt it::

    python tools/local_check.py --handshake-only --output results/handshake
    python tools/local_check.py --output results/public
    python tools/local_check.py --case vector_ties_and_filter --output results/ties

Pass ``--python-exe`` to test another interpreter (for example inside the
built image).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protocol_check"))

import run as driver  # noqa: E402  (path set up above)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python-exe", default=sys.executable)
    known, forwarded = parser.parse_known_args(argv)
    command = json.dumps(
        [
            known.python_exe,
            str(ROOT / "src" / "minivectordb" / "__main__.py"),
            "--data-dir",
            "{data_dir}",
        ]
    )
    return driver.main(["--command", command] + forwarded)


if __name__ == "__main__":
    sys.exit(main())
