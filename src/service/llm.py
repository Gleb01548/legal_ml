from loguru import logger
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from src.service.model_config import model_config


system_prompt = "Ты ИИ"


class LLMInference:
    def __init__(self, model_name: dict):

        logger.info(f"Загрузка модели {model_name}")

        model_path = hf_hub_download(**model_config[model_name])
        self.llm = Llama(
            model_path=model_path, max_tokens=10_000, n_gpu_layers=-1, n_ctx=10_000, verbose=False
        )

    def generate_from_message(self, user_message, temperature=0.2):
        return self.llm.create_chat_completion(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            seed=10,
        )["choices"][0]["message"]["content"]

    def generate(self, chat, temperature=0.2):
        return self.llm.create_chat_completion(
            chat,
            temperature=temperature,
            seed=10,
            stream=False,
        )["choices"][0]["message"]["content"]

    def generate_stream(self, chat, temperature=0.2):
        return self.llm.create_chat_completion(
            chat,
            temperature=temperature,
            seed=10,
            stream=True,
        )

    def generate_prompt(self, prompt, temperature=0.2):
        return self.llm(prompt, temperature=temperature)["choices"][0]["text"]
