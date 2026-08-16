from __future__ import annotations

import argparse
import json
import time

from memory_repository import MemoryRepository
from semantic_memory import MemoryConsolidator, MemoryWorker, SemanticMemoryStore


def run_once(*, repository: MemoryRepository, worker_id: str) -> dict[str, object] | None:
    store = SemanticMemoryStore(repository.path)
    worker = MemoryWorker(store, worker_id=worker_id)
    return worker.run_once(MemoryConsolidator(store))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate episodic memory jobs.")
    parser.add_argument("--worker-id", default="memory-worker-cli")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args()
    repository = MemoryRepository()
    while True:
        result = run_once(repository=repository, worker_id=args.worker_id)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not args.loop or result is None:
            break
        time.sleep(max(0.1, args.interval_seconds))


if __name__ == "__main__":
    main()
