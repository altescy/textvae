import logging
import os
import warnings

from textvae.commands import main

if os.environ.get("TEXTVAE_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("TEXTVAE_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", FutureWarning)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


def run() -> None:
    main(prog="textvae")


if __name__ == "__main__":
    run()
