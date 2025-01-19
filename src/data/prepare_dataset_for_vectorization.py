import os

import pandas as pd
from tqdm import tqdm


path_files = "./data/interim/split_9111_dataset"


def stat(answers, counter_expert, counter_rating, counter_len_chat, author_in_chat):
    answers = list(answers)
    len_answers = len(answers)

    counter_len_chat.append(len_answers)
    try:
        counter_expert.extend([i["status"] for i in answers])
        counter_rating.extend([i["rating"] for i in answers])
        for i in answers:
            if i["status"] == "Автор вопроса":
                author_in_chat.append(True)
                break
        else:
            author_in_chat.append(False)
    except:
        print(answers)


def norm(answers, description):
    answers = list(answers)
    try:
        len_answ = len(answers)
        rating = answers[0]["rating"]
        text = answers[0]["text"]
        len_description = len(description)
        len_text = len(text)
    except:
        len_answ, rating, text, len_description, len_text = None, None, None, None, None
    return len_answ, rating, text, len_description, len_text


def find_answer_with_max_score(answers):
    max_index = 0
    max_rating = 0
    for index, i in enumerate(answers):
        if max_rating < i["rating"]:
            max_index = index
            max_rating = i["rating"]
    return answers[max_index], max_rating


def prepare_df_for_vectorization(df: pd.DataFrame) -> pd.DataFrame:
    counter_expert = []
    counter_rating = []
    counter_len_chat = []
    author_in_chat = []

    for i in df["answers"].to_list():
        stat(i, counter_expert, counter_rating, counter_len_chat, author_in_chat)

    counter_len_chat_series = pd.Series(
        [i if i in [1, 2, 3, 4, 5] else -1 for i in counter_len_chat]
    )

    df["author_in_chat"] = author_in_chat
    df["counter_len_chat"] = counter_len_chat
    df["counter_len_chat_bins"] = counter_len_chat_series

    df_short = df[(~df["author_in_chat"]) & (df["counter_len_chat"] < 6)]
    df_short[["answers_max", "max_rating"]] = df_short.apply(
        lambda x: find_answer_with_max_score(x["answers"]), result_type="expand", axis=1
    )

    return df_short


bad_records = {
    "id": 18536837,
    "title": "Как мне получить субсидии на ком. платежи, если я ухаживаю за пожилым человеком старше 80 лет.",
    "description": "Как мне получить субсидии на ком. платежи, если я ухаживаю за пожилым человеком старше 80 лет.",
    "answers": [
        [
            {
                "user_name": "Парфенов В.Н.",
                "status": "Юрист",
                "rating": 4.6,
                "text": "К сожалению, то что вы ухаживаете за пожилым человеком старше 80 лет.-не является основанием для получения субсидии на оплату коммунальных услуг стст 153-155 ЖК РФ.",
            }
        ]
    ],
}


def prepare_dataset_for_vectorization(
    path_files_load: str, path_files_save: str
) -> None:

    for index, name_file in tqdm(enumerate(os.listdir(path_files_load))):
        df = pd.read_parquet(os.path.join(path_files_load, name_file))

        if not index:
            df_bad_record = pd.DataFrame(bad_records)
            df = pd.concat([df, df_bad_record])

        df = prepare_df_for_vectorization(df)
        df[["description", "answers_max", "id"]].to_parquet(
            os.path.join(path_files_save, f"question_aswers_{index}.parquet")
        )
