import argparse
import logging
import os

from dotenv import load_dotenv

from bee_bug_hunter.config import DEFAULT_LOG_FILE, DEFAULT_LOG_LEVEL, DEFAULT_MANIFEST
from bee_bug_hunter.logging_config import configure_logging
from bee_bug_hunter.orchestrator import monitor_loop


def main():
    load_dotenv()
    log_file = os.getenv("LOG_FILE", DEFAULT_LOG_FILE)
    configure_logging(
        level=getattr(logging, os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(), logging.INFO),
        log_file=log_file if log_file.lower() != "none" else None,
    )

    parser = argparse.ArgumentParser(description="Monitor a batch of API flows for bugs and performance issues.")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to the flows manifest YAML (list of flows + containers, poll interval).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the batch a single time and exit, instead of polling forever.",
    )
    args = parser.parse_args()

    monitor_loop(args.manifest, once=args.once)


if __name__ == "__main__":
    main()
