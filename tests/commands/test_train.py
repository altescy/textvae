from pathlib import Path

from textvae.commands import create_subcommand


def test_train_command(tmp_path: Path) -> None:
    workdir = tmp_path
    config_filename = workdir / "config.json"
    archive_filename = workdir / "archive.pkl"

    app = create_subcommand()
    args = app.parser.parse_args(
        [
            "train",
            "tests/fixtures/configs/textvae.json",
            "--workdir",
            str(workdir),
        ]
    )

    app(args)

    assert config_filename.is_file()
    assert archive_filename.is_file()
