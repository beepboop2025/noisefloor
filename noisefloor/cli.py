"""Offline JSON assessment: noisefloor market|narrative|capabilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .mcp_server import call_tool, capabilities
from .schemas import MAX_BODY_BYTES, dumps, loads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("market", "narrative", "capabilities"))
    parser.add_argument("path", nargs="?", default="-", help="JSON file, or - for stdin")
    parser.add_argument("--input", dest="input_path", help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            if args.path != "-" or args.input_path is not None:
                raise ValueError("capabilities takes no input")
            result = capabilities()
        else:
            if args.input_path is not None and args.path != "-":
                raise ValueError("provide a positional path or --input, not both")
            path = args.input_path if args.input_path is not None else args.path
            if path == "-":
                raw = sys.stdin.buffer.read(MAX_BODY_BYTES + 1)
            else:
                with Path(path).open("rb") as stream:
                    raw = stream.read(MAX_BODY_BYTES + 1)
            result = call_tool({"market": "market_assessment", "narrative": "narrative_triage"}[args.command], loads(raw))
        print(dumps(result))
        return 0
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
        print(dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
