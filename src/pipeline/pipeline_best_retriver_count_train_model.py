import os
from pathlib import Path

import click
import mlflow
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.find_retriver_count_train_model import train_model


os.environ["MLFLOW_EXPERIMENT_NAME"] = "best_retriver_count"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5000"


@click.command()
@click.option(
    "--path_load_data",
    type=click.Path(),
    default="./data/interim/data_param_tunning/data_records.parquet",
)
@click.option(
    "--path_save_model",
    type=click.Path(),
    default="./models",
)
@click.option(
    "--model_name",
    type=click.STR,
    default="",
)
@click.option(
    "--num_data",
    type=click.STR,
    default="",
)
def main(path_load_data, path_save_model, model_name, num_data):
    Path(path_save_model).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path_load_data)
    num_data = num_data.split()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    columns = [i for i in df.columns if "param" in i]

    mlflow.start_run()

    for col in tqdm(columns):
        for num in num_data:
            param = col.removeprefix("param_")
            use_rerank = param.endswith("_rerank")
            retriver_type = param.removesuffix("_rerank")

            params = {"retriver_type": retriver_type, "use_rerank": use_rerank, "num_data": num}
            mlflow.log_params(params)

            train = df.loc[df["cat"] == "train", col].to_list()[:num]
            valid = df.loc[df["cat"] == "valid", col].to_list()

            train = [
                tokenizer.apply_chat_template(i, tokenize=False, add_generation_prompt=True)
                for i in train
            ]
            valid = [
                tokenizer.apply_chat_template(i, tokenize=False, add_generation_prompt=True)
                for i in valid
            ]

            path_model_save_lora = os.path.join(path_save_model, f"{col}_{num}_lora")

            train_model(
                model_name=model_name,
                train=train,
                valid=valid,
                path_model_save_lora=path_model_save_lora,
            )

    mlflow.end_run()
