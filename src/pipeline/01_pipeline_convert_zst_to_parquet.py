import os
from pathlib import Path

import click

from src.data.convert_zst_to_perquet_9111_df import convert_zst_to_parquet


@click.command()
@click.option("--zst_path", type=click.Path(), default="./data/raw/questions.json.zst")
@click.option(
    "--path_save",
    type=click.Path(),
    default="./data/interim/split_9111_dataset/split_9111_dataset",
)
def main(zst_path, path_save):
    Path(os.path.dirname(path_save)).mkdir(parents=True, exist_ok=True)
    convert_zst_to_parquet(zst_path, path_save)


if __name__ == "__main__":
    main()
