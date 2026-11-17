"""Entry point: ``python -m minivectordb --data-dir /data``."""

import argparse
import os
import sys

if __package__ in (None, ""):  # allow ``python src/minivectordb/__main__.py``
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from minivectordb.server import run
else:
    from .server import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="MiniVectorDB stdin/stdout server")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("VECTORDB_DATA_DIR", "/data"),
        help="directory holding tables, catalog and commit log",
    )
    arguments = parser.parse_args(argv)
    return run(arguments.data_dir)


if __name__ == "__main__":
    sys.exit(main())
