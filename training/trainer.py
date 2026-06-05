# training/trainer.py

import os
from typing import Optional
from config import ModelConfig, TrainingConfig, DataConfig, NanoMindConfig


class NanoMindTrainer:
    """
    LoRA fine-tuning using Unsloth for memory efficiency.
    Designed to run on Kaggle free tier (T4 GPU, 16GB VRAM).
    """

    def __init__(self, config: NanoMindConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load base model with 4-bit quantization via Unsloth."""
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.config.model.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.model_name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=self.config.model.dtype,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        print("Base model loaded.")
        return self.model, self.tokenizer

    def apply_lora(self):
        """Attach LoRA adapters to the model."""
        from unsloth import FastLanguageModel

        print("Applying LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.model.lora_r,
            target_modules=self.config.model.target_modules,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.training.seed,
        )

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")
        return self.model

    def build_trainer(self, dataset):
        """Set up the SFTTrainer with training config."""
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import is_bfloat16_supported
        from datasets import Dataset

        print("Building trainer...")
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            optim=self.config.training.optimizer,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            seed=self.config.training.seed,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
        )

    
        if hasattr(dataset, 'samples'):
            dataset = Dataset.from_list(list(dataset.samples))
        elif isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset.samples,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            args=training_args,
            packing=False,
        )
        return self.trainer

    def train(self) -> dict:
        """Run training and return loss history."""
        if self.trainer is None:
            raise RuntimeError("Call build_trainer() before train()")

        print("Starting training...")
        result = self.trainer.train()

        metrics = {
            "train_loss": result.training_loss,
            "train_steps": result.global_step,
            "train_runtime": result.metrics.get("train_runtime", 0),
        }
        print(f"Training complete. Loss: {metrics['train_loss']:.4f}")
        return metrics

    def save(self, path: Optional[str] = None):
        """Save LoRA adapters only (much smaller than full model)."""
        save_path = path or self.config.training.output_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def run(self, dataset) -> dict:
        """Full training pipeline: load → LoRA → train → save."""
        self.load_model()
        self.apply_lora()
        self.build_trainer(dataset)
        metrics = self.train()
        self.save()
        return metrics