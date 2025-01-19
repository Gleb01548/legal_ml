import datasets
import pandas as pd
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer
from qdrant_client import QdrantClient, models

from src.conf import url_qdrant


def create_records(tokenizer, system_prompt, question, answer, context):
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": question,
        },
        {"role": "assistant", "content": answer},
    ]
    record = {}
    record["prompt"] = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=False
    )
    record["prompt_not_answer"] = tokenizer.apply_chat_template(
        prompt[:-1], tokenize=False, add_generation_prompt=True
    )
    record["answer"] = answer
    record["question"] = question

    if context:
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": question,
            },
            {"role": "system", "content": context},
            {"role": "assistant", "content": answer},
        ]
        record["prompt_context"] = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=False
        )
        record["prompt_context_not_answer"] = tokenizer.apply_chat_template(
            prompt[:-1], tokenize=False, add_generation_prompt=True
        )
        record["context"] = context
    else:
        record["prompt_context"] = None
        record["prompt_context_not_answer"] = None
        record["context"] = None

    return record


def create_prompt_dataset(
    collection_name: str,
    num_examples_for_train: int,
    rating_more: float,
    model_name: str,
    system_prompt: str,
    path_save_dataset: str,
):
    client = QdrantClient(url=url_qdrant)

    logger.info(
        f"Выгрузка данных. Кол-во экземпляров для выгрузки: {num_examples_for_train} "
        f"Рейтинг больше или равно {rating_more}"
    )
    points = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="rating",
                    range=models.Range(gte=rating_more),
                )
            ]
        ),
        limit=num_examples_for_train,
        with_vectors=True,
        timeout=300,
    )
    logger.info(f"Кол-во выгруженных записей {len(points[0])}")

    vectors_sample = [i.vector for i in points[0]]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Формирование контекста")
    payloads = []
    for vector in tqdm(vectors_sample):
        vect_search = client.query_points(
            collection_name=collection_name, query=vector, limit=5
        ).points

        payload, vect_search = vect_search[0].payload, vect_search[1:]
        if payload["question"] == vect_search[0].payload["question"]:
            vect_search = vect_search[1:]
        else:
            vect_search = vect_search[:1]

        prav_sit = (
            "Перед тем как дать консультацию изучи вопросы других людей "
            "и ответы на них других юристов:\n"
        )
        for i in vect_search:
            prav_sit += f"""Вопрос гражданина:
        {i.payload["question"]}
        Ответ юриста:
        {i.payload["answer"]}
        """
        payloads.append(
            {
                "system_prompt": system_prompt,
                "question": payload["question"],
                "answer": payload["answer"],
                "context": prav_sit,
            }
        )

    logger.info("Формирование промптов")
    dataset = []
    for record in tqdm(payloads):
        dataset.append(create_records(tokenizer, **record))

    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0)

    len_valid_data = int(len(dataset) / 100 * 15)
    train_data, valid_data = dataset[len_valid_data:], dataset[:len_valid_data]

    logger.info("Сохранение датасета")
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(train_data),
            "valid": datasets.Dataset.from_pandas(valid_data),
        }
    )

    dataset.save_to_disk(path_save_dataset)
