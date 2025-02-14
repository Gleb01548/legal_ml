import os
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
from src.features.create_prompt import create_prompt


@click.command()
@click.option("--model_name", type=click.STRING, default="unsloth/Qwen2.5-7B-Instruct")
@click.option("--limit", type=click.INT, default=3)
@click.option(
    "--system_prompt",
    type=click.STRING,
    default=(
        "Ты юрист и эксперт по российскому праву, твоя задача помогать людям. "
        "Изучи вопрос пользователя, изучи переданные затем тебе вопросы других "
        "граждан и ответы, которые давали на них другие юристы. "
        "На основании этих данных ответь на вопрос гражданина. "
        "Ответ пиши только на РУССКОМ языке!"
    ),
)
@click.option(
    "--path_load_dataset",
    type=click.STRING,
    default="./data/interim/data_param_tunning/data_records.parquet",
)
@click.option(
    "--path_save_dataset",
    type=click.STRING,
    default="./data/interim/data_param_tunning/data_prompts.parquet",
)
def main(path_load_dataset, path_save_dataset, model_name, system_prompt, limit):
    Path(os.path.dirname(path_save_dataset)).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path_load_dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    columns = [i for i in df.columns if i.startswith("param_")]

    logger.info(f"Колонки для промптов: {columns}")

    for col in tqdm([i for i in df.columns if i.startswith("param_")]):
        prompts = []
        for record in df[col].to_list():
            prompts.append(
                create_prompt(
                    record=record, system_prompt=system_prompt, tokenizer=tokenizer, limit=limit
                )
            )
        df[f"prompt_{col.removeprefix('param_')}"] = prompts
        df[f"len_{col.removeprefix('param_')}"] = [len(tokenizer.tokenize(i)) for i in prompts]

    columns = [i for i in df.columns if not i.startswith("param_")]

    logger.info("Сохраняем датасет.")

    df[columns].to_parquet(path_save_dataset, index=False)


if __name__ == "__main__":
    main()