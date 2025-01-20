import os

import mlflow
import datasets
from loguru import logger
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_name = "unsloth/Qwen2.5-7B-Instruct"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "trainer-mlflow-demo"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5000"


def train_model(
    model_name, path_dataset, path_model_save_vllm, path_model_save_gguf, path_model_save_lora
):
    max_seq_length = 2048
    r = 8
    use_rslora = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    dataset = datasets.load_from_disk(path_dataset)
    train = dataset["train"]
    valid = dataset["valid"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=use_rslora,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=valid,
        dataset_text_field="prompt_context",
        max_seq_length=max_seq_length,
        data_collator=collator,
        dataset_num_proc=16,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=5,  # Set this for 1 full training run.
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=0.5e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            overwrite_output_dir=True,
            load_best_model_at_end=True,
        ),
    )

    trainer_stats = trainer.train()
    logger.info(f"Статистика обучения: {trainer_stats}")
    mlflow.end_run()

    model.save_pretrained_merged(
        path_model_save_vllm,
        tokenizer,
        save_method="merged_16bit",
    )
    model.save_pretrained_merged(
        path_model_save_lora,
        tokenizer,
        save_method="lora",
    )
