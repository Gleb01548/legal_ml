import os
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm
from loguru import logger
from datasets import Dataset


from src.models.find_retriver_count_train_model import train_model


@click.command()
@click.option(
    "--path_load_data",
    type=click.Path(),
    default="./data/interim/data_param_tunning/data_prompts.parquet",
)
@click.option(
    "--path_save_models",
    type=click.Path(),
    default="./models/qwen/exp_retriver",
)
@click.option(
    "--model_name",
    type=click.STRING,
    default="unsloth/Qwen2.5-7B-Instruct",
)
@click.option(
    "--experiment_name",
    type=click.STRING,
    default="find_best_retriver",
)
def main(path_load_data, path_save_models, model_name, experiment_name):
    Path(path_save_models).mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка датасета")
    df = pd.read_parquet(path_load_data)

    columns = [i for i in df.columns if i.startswith("prompt")]
    logger.info(f"Ретриверы для перебора {columns}")

    for col in tqdm(columns):
        param = col.removeprefix("param_")
        use_rerank = param.endswith("_rerank")
        retriver_type = param.removesuffix("_rerank")

        train = Dataset.from_dict({"text": df.loc[df["cat"] == "train", col].to_list()})
        valid = Dataset.from_dict({"text": df.loc[df["cat"] == "valid", col].to_list()})

        params = {"retriver_type": retriver_type, "use_rerank": use_rerank, "num_train": len(train)}

        path_model_save_lora = os.path.join(path_save_models, f"{col}_lora")

        train_model(
            model_name=model_name,
            train=train,
            valid=valid,
            path_model_save_lora=path_model_save_lora,
            param_log=params,
            experiment_name=experiment_name,
        )


if __name__ == "__main__":
    main()
