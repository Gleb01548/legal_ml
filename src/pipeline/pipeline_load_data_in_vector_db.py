import os
from pathlib import Path
from datetime import datetime

import click
from src.data.load_data_in_vector_db import load_dataset_in_vector_db


@click.command()
@click.option(
    "--model_emb", type=click.STRING, default="BAAI/bge-m3"
)
@click.option("--qdrant_url", type=click.STRING, default="http://localhost:6333")
@click.option(
    "--path_load_data",
    type=click.Path(),
    default="./data/interim/question_answer_dataset",
)
@click.option("--collection_name", type=click.STRING, default="questions")
@click.option("--batch_size", type=click.INT, default=64)
@click.option(
    "--path_save_time",
    type=click.Path(),
    default="./data/interim/save_time_load_dataset_in_vector_db/",
)
@click.option("--min_rating", type=click.FLOAT, default=None)
def main(
    model_emb,
    qdrant_url,
    path_load_data,
    collection_name,
    batch_size,
    path_save_time,
    min_rating
):
    Path(path_save_time).mkdir(parents=True, exist_ok=True)

    load_dataset_in_vector_db(
        model_emb=model_emb,
        qdrant_url=qdrant_url,
        path_load_data=path_load_data,
        collection_name=collection_name,
        batch_size=batch_size,
        min_rating=min_rating
    )

    file_name = (
        str(datetime.now())
        .replace("-", "_")
        .replace(" ", "T")
        .replace(":", "")
        .replace(".", "")
    )

    path_save_time = os.path.join(path_save_time, file_name)
    open(path_save_time, "w+").close()


if __name__ == "__main__":
    main()