import re

import gradio as gr
from loguru import logger
from src.service.llm import LLMInference
from src.service.rag import Rag
from langchain import PromptTemplate


llm = LLMInference(model_name="r1")
think_pattern = r"<think>(.*?)</think>"

rag = Rag(
    collection_name="911_hybrid_rating_points",
    qdrant_url="http://localhost:6333",
    embedding_model="BAAI/bge-m3",
    retriver_types="dense_sparse_query",
    reranker=None,
    limit=5,
)
system_template = """
**Роль**
Ты юрист и эксперт по российскому праву, твоя задача помогать людям.
Ответ пиши только на РУССКОМ языке!
Если тебя спрашиваю о том, кто ты, то отвечай, что ты ИИ для юридических вопросов.
При даче ответа на вопрос не опирайся на обстрактное знание

**Задача**
Твоя задача ответить на вопрос пользователя. Прежде чем давать ответ на
вопрос пользователя изучи переданный тебе контекст. Контекст состоит из
вопросов других граждан и ответов на них других граждан.

**Контекст**
{context}
"""


def predict(message, history):
    history = []
    print("message", message)
    print("history", history)
    context = rag.create_context(user_message=message)
    system_prompt = (
        PromptTemplate(
            template=system_template,
            partial_variables={"user_message": message, "context": context},
        )
        .format_prompt()
        .text
    )

    answer = llm.generate(
        chat=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": message},
        ]
    )
    logger.info(f"Ответ модели: {answer}")
    final_response = re.sub(think_pattern, "", answer, flags=re.DOTALL).strip()
    final_response = f"""
Контекст
{context}

Ответ модели
{answer}
    """
    return final_response


demo = gr.ChatInterface(predict, type="messages")

demo.launch(share=False)
