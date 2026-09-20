import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retrieval.common.base import DENSE_OUTPUT_DIRECTORY, build_dense_index


if __name__ == "__main__":
    try:
        build_dense_index(DENSE_OUTPUT_DIRECTORY)
    except Exception as error:
        print()
        print("Index creation failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise SystemExit(1)
