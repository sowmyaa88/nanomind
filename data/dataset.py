# data/dataset.py

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from config import DataConfig


class AlpacaDataset(Dataset):
    """
    Loads and formats the Alpaca-cleaned dataset
    into instruction-following prompt/response pairs.
    """

    def __init__(
        self,
        config: DataConfig,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.samples = self._load_and_format(split, max_samples)

    def _load_and_format(self, split: str, max_samples: Optional[int]) -> List[Dict]:
        """Load dataset and format into prompt strings."""
        print(f"Loading dataset: {self.config.dataset_name}")
        raw = load_dataset(self.config.dataset_name, split=self.config.split)

        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        samples = []
        for row in raw:
            prompt = self._format_prompt(
                instruction=row.get(self.config.instruction_key, ""),
                input_text=row.get(self.config.input_key, ""),
                output=row.get(self.config.output_key, ""),
            )
            if prompt:
                samples.append({"text": prompt})

        print(f"Loaded {len(samples)} samples")
        return samples

    def _format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format a single sample into Alpaca prompt format."""
        if not instruction:
            return ""

        if input_text.strip():
            input_section = (
                f", paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Input:\n{input_text}\n\n"
            )
        else:
            input_section = (
                ". Write a response that appropriately "
                "completes the request.\n\n"
            )

        return (
            f"Below is an instruction that describes a task"
            f"{input_section}"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def get_raw_texts(self) -> List[str]:
        """Return all formatted texts as a list."""
        return [s["text"] for s in self.samples]


def get_eval_prompts(config: DataConfig, n: int = 50) -> List[Dict]:
    """
    Load a small set of samples for evaluation.
    Returns list of dicts with instruction, input, expected output.
    """
    raw = load_dataset(config.dataset_name, split=config.split)
    # Take from the end of dataset to avoid overlap with training samples
    eval_start = len(raw) - n
    eval_samples = raw.select(range(eval_start, len(raw)))

    prompts = []
    for row in eval_samples:
        instruction = row.get(config.instruction_key, "")
        input_text = row.get(config.input_key, "")
        expected = row.get(config.output_key, "")

        if not instruction:
            continue

        if input_text.strip():
            prompt = (
                f"Below is an instruction that describes a task, paired with "
                f"an input that provides further context. Write a response that "
                f"appropriately completes the request.\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. Write a response "
                f"that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        prompts.append({
            "instruction": instruction,
            "input": input_text,
            "expected": expected,
            "prompt": prompt,
        })

    return prompts