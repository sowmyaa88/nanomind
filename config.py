# config.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Base model and LoRA configuration."""
    model_name: str = "Qwen/Qwen2-0.5B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          
    dtype: Optional[str] = None        

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    output_dir: str = "outputs/nanomind-finetuned"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    fp16: bool = False
    bf16: bool = False                 
    max_steps: int = 100               


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "yahma/alpaca-cleaned"
    split: str = "train"
    max_samples: int = 2000            
    test_size: float = 0.1
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"

    # Alpaca prompt template
    prompt_template: str = (
        "Below is an instruction that describes a task"
        "{input_section}"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    num_eval_samples: int = 50
    gemini_model: str = "gemini-2.0-flash"
    score_key: str = "score"
    max_score: int = 10


@dataclass
class NanoMindConfig:
    """Master config — pass this around everywhere."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    project_name: str = "nanomind"
    run_name: str = "qwen-0.5b-alpaca-lora"



DEFAULT_CONFIG = NanoMindConfig()