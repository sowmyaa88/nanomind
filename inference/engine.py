# inference/engine.py

import torch
from typing import List, Dict, Optional
from config import ModelConfig, InferenceConfig, NanoMindConfig


class InferenceEngine:
    """
    Runs inference on a fine-tuned or base model.
    Supports both standard HuggingFace and Unsloth-optimized inference.
    """

    def __init__(self, config: NanoMindConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_base(self):
        """Load the original base model (before fine-tuning) for comparison."""
        from unsloth import FastLanguageModel

        print(f"Loading base model: {self.config.model.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.model_name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=self.config.model.dtype,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        print("Base model ready for inference.")
        return self.model, self.tokenizer

    def load_finetuned(self, adapter_path: Optional[str] = None):
        """Load the fine-tuned model with LoRA adapters."""
        from unsloth import FastLanguageModel

        path = adapter_path or self.config.training.output_dir
        print(f"Loading fine-tuned model from: {path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=self.config.model.max_seq_length,
            dtype=self.config.model.dtype,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        print("Fine-tuned model ready for inference.")
        return self.model, self.tokenizer

    def generate(self, prompt: str) -> str:
        """Generate a response for a single prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load a model first with load_base() or load_finetuned()")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference.max_new_tokens,
                temperature=self.config.inference.temperature,
                do_sample=self.config.inference.do_sample,
                top_p=self.config.inference.top_p,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  Generating {i+1}/{len(prompts)}...", end="\r")
            results.append(self.generate(prompt))
        print()
        return results

    def run_eval_set(self, eval_samples: List[Dict]) -> List[Dict]:
        """
        Run inference on evaluation samples.
        Each sample must have a 'prompt' key.
        Returns samples with 'generated' field added.
        """
        print(f"Running inference on {len(eval_samples)} samples...")
        prompts = [s["prompt"] for s in eval_samples]
        outputs = self.generate_batch(prompts)

        results = []
        for sample, output in zip(eval_samples, outputs):
            results.append({
                **sample,
                "generated": output,
            })
        return results