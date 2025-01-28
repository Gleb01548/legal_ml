import os
import time

import pandas as pd
from tqdm import tqdm
from loguru import logger
from qdrant_client import QdrantClient, models

from src.utils.embeddnig_api import EmbeddingAPI


def load_df_in_vector_db(
    df: pd.DataFrame,
    embedding_api: EmbeddingAPI,
    client: QdrantClient,
    batch_size: int,
    collection_name: str,
    min_rating: float,
) -> None:
    index0 = 0

    if min_rating:
        len_df = len(df)
        df["rating"] = [i["rating"] for i in df["answers_max"].to_list()]
        df = df[df["rating"] > min_rating]
        len_df_past = len(df)
        count_droped = len_df - len_df_past

        logger.info(
            (f"Было записей: {len_df:,}. "
             f"Записей после фильтра с минимальным рейтингом {min_rating}: {len_df_past:,}. "
             f"Всего удалено записей: {count_droped:,}")
        )

    len_df = len(df)

    for index1 in tqdm(range(batch_size, len_df + batch_size, batch_size)):
        batch = df.iloc[index0:index1].to_dict(orient="records")

        index0 = index1

        batch = [
            {
                "id": i["id"],
                "question": i["description"],
                "answer": i["answers_max"]["text"],
                "rating": i["answers_max"]["rating"],
            }
            for i in batch
        ]

        for _ in range(10):
            try:
                embeddings_question = embedding_api.get_embeddings([i["question"] for i in batch])
                break
            except:
                time.sleep(3)

        batch_points = [
            models.PointStruct(
                id=record["id"],
                payload={
                    "question": record["question"],
                    "answer": record["answer"],
                    "rating": record["rating"],
                },
                vector={
                    "dense": dence,
                    # "colbert": colbert,
                    "text-sparse": models.SparseVector(
                        indices=list(sparce.keys()),
                        values=list(sparce.values()),
                    ),
                },
            )
            for dence, sparce, record in zip(
                embeddings_question["dense_vecs"],
                embeddings_question["lexical_weights"],
                # embeddings_question["colbert_vecs"],
                batch,
            )
        ]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch_points,
            )
        except:
            time.sleep(10)
            logger.error("Ошибка при загрузке данных в Qdrant")


def load_dataset_in_vector_db(
    model_emb: str,
    qdrant_url: str,
    path_load_data: str,
    collection_name: str,
    batch_size: int,
    min_rating: float,
):
    logger.info("Инициализация api эмбеддиг модели и клиента qdrant")
    embedding_api = EmbeddingAPI(model_emb)
    client = QdrantClient(url=qdrant_url)
    client.delete_collection(collection_name, timeout=1000)

    logger.info(f"Создание коллекции с именем {collection_name}")

    vectors = embedding_api.get_embeddings("test1")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=vectors["dense_vecs"].shape[0], distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={"text-sparse": models.SparseVectorParams()},
        timeout=1000,
    )

    logger.info(f"Начат процесс векторизации данных и их загрузки в коллекцию {collection_name}")
    for name_file in tqdm(os.listdir(path_load_data)):
        df = pd.read_parquet(os.path.join(path_load_data, name_file))
        load_df_in_vector_db(
            df=df,
            embedding_api=embedding_api,
            client=client,
            batch_size=batch_size,
            collection_name=collection_name,
            min_rating=min_rating,
        )
    logger.info("Данные в коллекцию успешно загружены")
    logger.info("Проверяем статус коллекции")
    """
    Каждые 3 секунды будет проверятся статус коллекции, если код статуса green или red,
    то вернет статус код. Всего попыток 100.

    По истечении всех попыток вернет последний код статуса.
    """
    try:
        for _ in range(200):
            status = client.get_collection(collection_name=collection_name).status
            if status in ["green", "red"]:
                return status
            elif status == "yellow":
                pass
            elif status == "grey":
                logger.info(
                    f"collection_name: {collection_name}. Код статуса grey. Запуск update_collection"
                )
                client.update_collection(
                    collection_name=collection_name,
                    optimizer_config=models.OptimizersConfigDiff(),
                )

            time.sleep(3)
        return status
    except Exception as e:
        logger.error(f"Ошибка при получении статуса коллекции: {e}")
        raise
