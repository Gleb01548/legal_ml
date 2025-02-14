from pathlib import Path

import click
from src.data.prepare_dataset_for_vectorization import prepare_dataset_for_vectorization


@click.command()
@click.option("--path_load", type=click.Path(), default="./data/interim/split_9111_dataset")
@click.option("--path_save", type=click.Path(), default="./data/interim/question_answer_dataset")
def main(path_load, path_save):
    Path(path_save).mkdir(parents=True, exist_ok=True)
    prepare_dataset_for_vectorization(path_load, path_save)


if __name__ == "__main__":
    main()
