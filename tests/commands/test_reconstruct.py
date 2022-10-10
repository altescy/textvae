from pathlib import Path

from textvae.commands import create_subcommand


def test_reconstruct_command(tmp_path: Path) -> None:
    archive_filename = "tests/fixtures/models/archive.pkl"
    input_filename = "tests/fixtures/data/sentences.txt"
    output_filename = tmp_path / "reconstructions.txt"

    app = create_subcommand()
    args = app.parser.parse_args(
        [
            "reconstruct",
            str(archive_filename),
            "--input-filename",
            str(input_filename),
            "--output-filename",
            str(output_filename),
            "--device",
            "cpu",
        ]
    )

    app(args)

    assert output_filename.is_file()
