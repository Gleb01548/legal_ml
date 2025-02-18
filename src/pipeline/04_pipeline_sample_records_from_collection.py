import os
from pathlib import Path

import click
import pandas as pd

from src.data.sample_records_from_collection import sample_records_from_collection


@click.command()
@click.option(
    "--path_save", type=click.Path(), default="./data/interim/data_param_tunning/points.csv"
)
@click.option("--qdrant_url", type=click.STRING, default="http://localhost:6333")
@click.option("--collection_name", type=click.STRING, default="./data/interim/data_param_tunning/")
@click.option("--num_points", type=click.INT, default=10_000)
def main(path_save, qdrant_url, collection_name, num_points):
    Path(os.path.dirname(path_save)).mkdir(parents=True, exist_ok=True)
    points = sample_records_from_collection(
        url_qdrant_client=qdrant_url, collection_name=collection_name, num_points=num_points
    )
    len_valid_data = int(len(points) / 100 * 20)
    cat = ["train"] * (len(points) - len_valid_data) + ["valid"] * len_valid_data
    pd.DataFrame({"points": points, "cat": cat}).to_csv(path_save, index=False)


if __name__ == "__main__":
    main()
