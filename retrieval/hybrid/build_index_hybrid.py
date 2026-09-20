import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retrieval.common.base import HYBRID_OUTPUT_DIRECTORY, build_hybrid_index


if __name__ == "__main__":
    try:
        build_hybrid_index(HYBRID_OUTPUT_DIRECTORY)
    except Exception as error:
        print()
        print("Hybrid index creation failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise SystemExit(1)
