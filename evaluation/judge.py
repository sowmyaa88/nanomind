# evaluation/judge.py

import os
import json
import time
from typing import List, Dict, Optional
from config import EvaluationConfig


JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI assistant's response.

You will be given:
- An instruction (what the user asked)
- An input (optional context)
- The model's response

Score the response from 0 to 10 based on:
- Relevance: Does it address the instruction?
- Accuracy: Is the information correct?
- Completeness: Is the response thorough?
- Clarity: Is it well-written and easy to understand?

Respond with ONLY a JSON object in this exact format:
{
  "score": <number 0-10>,
  "reasoning": "<one sentence explanation>"
}"""


class GeminiJudge:
    """
    Uses Gemini as an LLM-as-a-Judge to score model outputs.
    Compares base model vs fine-tuned model responses.
    """

    def __init__(self, config: EvaluationConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Pass it or set as env variable.")
        self._setup_client()

    def _setup_client(self):
        """Initialize Gemini client."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.config.gemini_model)
        print(f"Gemini judge ready: {self.config.gemini_model}")

    def score_single(
        self,
        instruction: str,
        input_text: str,
        response: str,
    ) -> Dict:
        """Score a single response. Returns dict with score and reasoning."""
        user_message = f"""### Instruction:
{instruction}

### Input:
{input_text if input_text.strip() else "(none)"}

### Response to evaluate:
{response}"""

        try:
            result = self.client.generate_content(
                f"{JUDGE_PROMPT}\n\n{user_message}"
            )
            raw = result.text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return {
                "score": float(parsed.get("score", 0)),
                "reasoning": parsed.get("reasoning", ""),
                "error": None,
            }

        except Exception as e:
            return {"score": 0.0, "reasoning": "", "error": str(e)}

    def evaluate_batch(
        self,
        samples: List[Dict],
        delay: float = 1.0,
    ) -> List[Dict]:
        """
        Score a list of samples.
        Each sample needs: instruction, input, generated.
        Returns samples with score and reasoning added.
        """
        print(f"Evaluating {len(samples)} samples with Gemini...")
        results = []

        for i, sample in enumerate(samples):
            print(f"  Scoring {i+1}/{len(samples)}...", end="\r")
            scored = self.score_single(
                instruction=sample.get("instruction", ""),
                input_text=sample.get("input", ""),
                response=sample.get("generated", ""),
            )
            results.append({**sample, **scored})
            time.sleep(delay)   # rate limiting

        print()
        return results

    def compare(
        self,
        base_results: List[Dict],
        finetuned_results: List[Dict],
    ) -> Dict:
        """
        Compare base model vs fine-tuned model scores.
        Returns summary statistics.
        """
        base_scores = [r["score"] for r in base_results if r.get("error") is None]
        ft_scores = [r["score"] for r in finetuned_results if r.get("error") is None]

        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
        ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else 0
        improvement = ft_avg - base_avg
        improvement_pct = (improvement / base_avg * 100) if base_avg > 0 else 0

        summary = {
            "base_model": {
                "avg_score": round(base_avg, 2),
                "min_score": round(min(base_scores), 2) if base_scores else 0,
                "max_score": round(max(base_scores), 2) if base_scores else 0,
                "n": len(base_scores),
            },
            "finetuned_model": {
                "avg_score": round(ft_avg, 2),
                "min_score": round(min(ft_scores), 2) if ft_scores else 0,
                "max_score": round(max(ft_scores), 2) if ft_scores else 0,
                "n": len(ft_scores),
            },
            "improvement": {
                "absolute": round(improvement, 2),
                "percent": round(improvement_pct, 1),
            },
        }

        print("\n=== Evaluation Summary ===")
        print(f"Base model avg score:       {base_avg:.2f} / 10")
        print(f"Fine-tuned model avg score: {ft_avg:.2f} / 10")
        print(f"Improvement:                +{improvement:.2f} ({improvement_pct:.1f}%)")
        print("==========================\n")

        return summary