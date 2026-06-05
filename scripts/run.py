# scripts/run.py

import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NanoMindConfig, DEFAULT_CONFIG
from data.dataset import AlpacaDataset, get_eval_prompts
from training.trainer import NanoMindTrainer
from inference.engine import InferenceEngine
from evaluation.judge import GeminiJudge


def run_pipeline(config: NanoMindConfig = DEFAULT_CONFIG):
    """
    Full NanoMind pipeline:
    1. Load + format dataset
    2. Fine-tune base model with LoRA
    3. Run inference — base model vs fine-tuned
    4. Score both with Gemini judge
    5. Print and save comparison results
    """

    print("\n" + "="*60)
    print("  NanoMind — Small Language Model Training Pipeline")
    print("="*60 + "\n")

    # ── 1. Load dataset ───────────────────────────────────────────
    print("── Phase 1: Loading Dataset ──")
    trainer = NanoMindTrainer(config)
    trainer.load_model()
    trainer.apply_lora()

    dataset = AlpacaDataset(
        config=config.data,
        tokenizer=trainer.tokenizer,
        max_samples=config.data.max_samples,
    )
    print(f"Dataset ready: {len(dataset)} training samples\n")

    # ── 2. Train ──────────────────────────────────────────────────
    print("── Phase 2: Fine-tuning ──")
    trainer.build_trainer(dataset)
    train_metrics = trainer.train()
    trainer.save()
    print(f"Training metrics: {train_metrics}\n")

    # ── 3. Load eval prompts ──────────────────────────────────────
    print("── Phase 3: Preparing Evaluation Set ──")
    eval_samples = get_eval_prompts(
        config=config.data,
        n=config.evaluation.num_eval_samples,
    )
    print(f"Eval set: {len(eval_samples)} samples\n")

    # ── 4. Base model inference ───────────────────────────────────
    print("── Phase 4: Base Model Inference ──")
    base_engine = InferenceEngine(config)
    base_engine.load_base()
    base_results = base_engine.run_eval_set(eval_samples)
    print()

    # ── 5. Fine-tuned model inference ─────────────────────────────
    print("── Phase 5: Fine-tuned Model Inference ──")
    ft_engine = InferenceEngine(config)
    ft_engine.load_finetuned()
    ft_results = ft_engine.run_eval_set(eval_samples)
    print()

    # ── 6. Evaluate with Gemini judge ─────────────────────────────
    print("── Phase 6: Gemini Judge Evaluation ──")
    judge = GeminiJudge(config.evaluation)

    print("Scoring base model outputs...")
    base_scored = judge.evaluate_batch(base_results)

    print("Scoring fine-tuned model outputs...")
    ft_scored = judge.evaluate_batch(ft_results)

    # ── 7. Compare and save results ───────────────────────────────
    print("── Phase 7: Results ──")
    summary = judge.compare(base_scored, ft_scored)

    # Save full results to JSON
    os.makedirs("outputs", exist_ok=True)
    results = {
        "config": {
            "model": config.model.model_name,
            "lora_r": config.model.lora_r,
            "max_steps": config.training.max_steps,
            "dataset": config.data.dataset_name,
            "train_samples": config.data.max_samples,
            "eval_samples": config.evaluation.num_eval_samples,
        },
        "train_metrics": train_metrics,
        "evaluation_summary": summary,
        "base_model_samples": base_scored[:5],      # save first 5 for README
        "finetuned_samples": ft_scored[:5],
    }

    output_path = "outputs/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Full results saved to {output_path}")
    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    run_pipeline()