import os
from pathlib import Path

import click

from src.models.train_model import train_model


@click.command()
@click.option("--model_name", type=click.STRING, default="unsloth/Qwen2.5-7B-Instruct")
@click.option("--path_dataset", type=click.Path(), default="data/processed/100k_6_qwen")
@click.option("--path_model_save", type=click.Path(), default="./models/qwen")
def main(model_name, path_dataset, path_model_save):
    path_model_save_vllm = os.path.join(path_model_save, "vllm")
    path_model_save_gguf = os.path.join(path_model_save, "gguf")
    path_model_save_lora = os.path.join(path_model_save, "lora")
    Path(path_model_save_vllm).mkdir(parents=True, exist_ok=True)
    Path(path_model_save_gguf).mkdir(parents=True, exist_ok=True)
    train_model(
        model_name=model_name,
        path_dataset=path_dataset,
        path_model_save_vllm=path_model_save_vllm,
        path_model_save_gguf=path_model_save_gguf,
        path_model_save_lora=path_model_save_lora
    )


if __name__ == "__main__":
    main()
