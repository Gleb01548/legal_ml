from pathlib import Path

import click
from src.features.create_prompt_dataset import create_prompt_dataset


@click.command()
@click.option("--collection_name", type=click.STRING, default="questions")
@click.option("--num_examples_for_train", type=click.INT, default=20_000)
@click.option("--rating_more", type=click.FLOAT, default=7)
@click.option("--model_name", type=click.STRING, default="unsloth/Qwen2.5-7B-Instruct")
@click.option(
    "--system_prompt",
    type=click.STRING,
    default=(
        "Ты юрист. Ты консультируешь людей по разным проблемам, внимательно прочитай, "
        "что тебе пишут, ответь на поставленные вопросы и дай консультацию. "
        "Ответ пиши только на РУССКОМ языке!"
    ),
)
@click.option("--path_save_dataset", type=click.STRING, default="./data/processed/100k_6_qwen")
def main(
    collection_name,
    num_examples_for_train,
    rating_more,
    model_name,
    system_prompt,
    path_save_dataset,
):
    Path(path_save_dataset).mkdir(parents=True, exist_ok=True)
    create_prompt_dataset(
        collection_name=collection_name,
        num_examples_for_train=num_examples_for_train,
        rating_more=rating_more,
        model_name=model_name,
        system_prompt=system_prompt,
        path_save_dataset=path_save_dataset,
    )


if __name__ == "__main__":
    main()
