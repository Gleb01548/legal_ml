import os
import io
import zstandard as zstd

import polars as pl
from tqdm import tqdm


def convert_zst_to_parquet(zst_path: str, path_save: str) -> None:
    buff = []
    problem_line = 0
    counter = 0
    counter_line = 0
    with open(zst_path, "rb") as file:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(file)
        stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for index, line in tqdm(enumerate(stream)):
            try:
                buff.append(eval(line))
                counter_line += 1
            except ValueError:
                print(line)
                problem_line += 1
            if counter_line == 1_000_000:
                df = pl.from_dicts(buff, strict=False)
                buff = []
                counter += 1
                counter_line = 0
                df.write_parquet(path_save + str(counter) + ".parquet")
        else:
            df = pl.from_dicts(buff, strict=False)
            buff = []
            counter += 1
            counter_line = 0
            df.write_parquet(path_save + str(counter) + ".parquet")

    print("Проблемных строк:", problem_line)
