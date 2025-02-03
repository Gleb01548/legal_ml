from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm
from FlagEmbedding import FlagLLMReranker

from src.features.create_data_records import CreateDataRecords


@click.command()
@click.option(
    "--path_points", type=click.Path(), default="./data/interim/data_param_tunning/points.csv"
)
@click.option("--collection_name", type=click.STRING, default="")
@click.option("--rerank_model", type=click.STRING, default="BAAI/bge-reranker-v2-gemma")
@click.option(
    "--qdrant_url",
    type=click.STRING,
    default="http://localhost:6333",
)
@click.option(
    "--path_save",
    type=click.STRING,
    default="./data/interim/data_param_tunning/data_records.parquet",
)
@click.option(
    "--params",
    type=click.STRING,
    default="""[
                {'retriver_type':'dense_query', 'use_rerank':False},
                {'retriver_type':'dense_query', 'use_rerank':True},
                {'retriver_type':'dense_sparse_query', 'use_rerank':False},
                {'retriver_type':'dense_sparse_query', 'use_rerank':True},
                ]""",
)
@click.option(
    "--limit",
    type=click.INT,
    default=3,
)
def main(path_points, collection_name, rerank_model, qdrant_url, path_save, params, limit):
    Path(path_save).mkdir(parents=True, exist_ok=True)

    params = eval(params)
    cdr = CreateDataRecords(
        reranker=FlagLLMReranker(rerank_model, device="cuda"), qdrant_url=qdrant_url
    )

    df = pd.read_csv(path_points)
    ids_points = df["points"].to_list()
    logger.info(f"Кол-во строк {len(ids_points)}")

    for param in tqdm(params):
        logger.info(f"Подготовка датасета с параметрами: {param}")
        records = cdr.create_data_records(
            collection_name=collection_name,
            ids=ids_points,
            retriver_type=param["retriver_type"],
            use_reranker=param["use_rerank"],
            limit=limit,
        )
        if param["use_rerank"]:
            suffix = "_rerank"
        else:
            suffix = ""
        df[f"{param['retriver_type']}{suffix}"] = records

    logger.info("Сохранение данных")
    df.to_parquet(path_save)


if __name__ == "__main__":
    main()
