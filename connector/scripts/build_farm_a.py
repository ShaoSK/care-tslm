"""Quick check on real data: convert only wind farm A of a local CARE_To_Compare extraction.

Usage (from the TimeNet repo root, after `make sync`):

    uv run python build_farm_a.py ~/data/CARE_To_Compare

It writes a TimeF version to ~/.timenet/care-farm-a-check and prints a summary. It does not use the
engine or the registry, so it is only a check; publish with `timenet-build build`.
"""

from pathlib import Path
import sys
import time

from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.connector import (
    CareToCompareConnector,
    CareWindFarm,
)


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    farms = [farm for farm in CareWindFarm.discover(root) if farm.name == "A"]
    started = time.time()
    dataset = CareToCompareConnector().convert(farms)
    print(f"converted wind farm A in {time.time() - started:.0f} s")
    version_dir = dataset.write(path=Path("~/.timenet/care-farm-a-check").expanduser())
    print(f"wrote {version_dir}")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        reader.read().describe()


if __name__ == "__main__":
    main()
