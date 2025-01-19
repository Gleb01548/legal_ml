import time

import docker
import requests
from loguru import logger

client = docker.from_env()


def up_embedding_model(embedding_service_url):
    container = client.containers.run(
        image="ghcr.io/huggingface/text-embeddings-inference:1.6",
        command="--model-id BAAI/bge-m3 --max-client-batch-size=1024 --max-batch-requests=1024 --max-batch-tokens=64000 --auto-truncate",
        ports={"80": "8080"},
        volumes=["/embedding_model_data:/data"],
        runtime="nvidia",
        detach=True,
        auto_remove=True,
    )
    logger.info("Поднятие докер контейнера с эмбеддинг моделью")
    for _ in range(20):
        try:
            res = requests.post(embedding_service_url, json={"inputs": ["test"]})
        except Exception as e:
            time.sleep(6)
            continue

        if res.status_code == 200:
            logger.info("Контейнер с эмбеддинг моделью успешно поднят!")
            return container
        else:
            time.sleep(6)

    logger.error("Проблемы при поднятии контейнера с эмбеддинг моделей")
    assert False, "Проблемы при поднятии контейнера с эмбеддинг моделей"
