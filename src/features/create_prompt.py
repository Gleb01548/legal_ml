from typing import List


def create_prompt(record: List[dict], system_prompt: str, tokenizer, limit: int = 3) -> str:
    context = record["context"][:limit]
    question = record["question_answer"]["question"]
    answer = record["question_answer"]["answer"]

    string_context = "Прежде чем дать ответ изучи вопросы других граждан и ответы на них юристов:"

    for i in context:
        string_context += (
            f"\n\nВопрос гражданина:\n{i['question']}\n\n" f"Ответ юриста:\n{i['answer']}"
        )

    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "system",
            "content": string_context,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
