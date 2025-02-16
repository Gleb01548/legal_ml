import mlflow
import torch
import pandas as pd
from transformers import TrainerCallback


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_examples=3, max_length=512):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_examples = num_examples
        self.max_length = max_length

    def on_evaluate(self, args, state, control, **kwargs):
        # Выбираем примеры для демонстрации
        samples = [self.eval_dataset[i] for i in range(self.num_examples)]

        # Извлекаем промпты и целевые ответы
        results = []
        for sample in samples:
            full_text = sample["text"]
            # Разделяем текст на промпт и ответ
            parts = full_text.split("<|im_start|>assistant\n")
            prompt = parts[0] + "<|im_start|>assistant\n"
            target = parts[1].split("<|im_end|>")[0] if len(parts) > 1 else ""

            # Генерируем ответ модели
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

            with torch.no_grad():
                outputs = self.trainer.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )

            generated = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], skip_special_tokens=True
            )

            results.append(
                {
                    "prompt": prompt.replace("<|im_start|>assistant\n", "").strip(),
                    "generated": generated,
                    "target": target.strip(),
                }
            )

        # Создаем и логируем таблицу
        df = pd.DataFrame(results)
        mlflow.log_table(df, artifact_file=f"generations/step_{state.global_step}.json")
