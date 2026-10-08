"""Identity of installed Python source files; not a source authenticity claim."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def implementation_sha256():
    """Hash filename + content hashes, captured once for this process.

    Immutable release directories make this comparable with a verified wheel.
    It excludes bytecode, build times, environment variables and caller inputs.
    """
    root = Path(__file__).resolve().parent
    rows = [p.name + ":" + hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
