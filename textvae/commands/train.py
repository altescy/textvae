import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict

import colt

from textvae.commands.subcommand import Subcommand


@Subcommand.register("train")
class TrainCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument("config", type=Path)
        self.parser.add_argument("-w", "--workdir", type=Path, default=None)
        self.parser.add_argument("--overrides", type=json.loads, default=None)

    def load_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        with open(args.config) as f:
            config = json.load(f)
            assert isinstance(config, dict)

        if args.overrides:

            def replace(config: Dict[str, Any], key: str, value: Any) -> None:
                if "." in key:
                    head, tail = key.split(".", 1)
                    if head in config:
                        replace(config[head], tail, value)
                elif key in config:
                    config[key] = value

            for key, value in args.overrides.items():
                replace(config, key, value)

        return config

    def save_config(self, config: Dict[str, Any], filename: Path) -> None:
        with open(filename, "w") as jsonfile:
            json.dump(config, jsonfile, indent=2, ensure_ascii=False)

    def get_workdir(self, args: argparse.Namespace) -> Path:
        if args.workdir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            workdir = Path("output") / timestamp
        else:
            workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def run(self, args: argparse.Namespace) -> None:
        from textvae.trainer import Trainer

        workdir = self.get_workdir(args)

        config = self.load_config(args)
        self.save_config(config, workdir / "config.json")

        trainer = colt.build(config, Trainer)
        trainer.train(workdir)
