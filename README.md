# Small Language Model

## Prerequisites
- Install all dependencies from requirements.txt using:
```bash
pip install -r requirements.txt
```
- Thirst to explore 😊

## Getting Started
A small language model is a compact AI language model with far fewer parameters (typically millions to a few billion) designed to perform targeted NLP tasks efficiently on limited hardware, trading broad capability for lower latency, cost, and easier deployment.

This repository has codes to experiment with different libraries and different techniques to make a high performing small language model. 

## Experiments
- **Experiment 1** has code with using Unsloth and LLMCompressor libraries. It also uses vLLM for faster inference. `Qwen.py` is made without any optimizations to set the baseline. All files use Alpaca-cleaned dataset.

- **Experiment 2** has code with using gpt-oss 20B model for Lora finetuning. Math dataset is used for evaluation from https://github.com/ziye2chen/DEMI-MathAnalysis. Gemini 2.5 pro is used as a LLM as a judge. 

## Modules Documentation

This directory contains modularized components for Small Language Model (SLM) training, inference, and evaluation. The architecture follows a clean separation of concerns with abstract base classes and specialized implementations.

### 📁 Directory Structure

```
Modules/
├── core/               # Base classes and interfaces
├── data/               # Data processing utilities
├── evaluation/         # Model evaluation tools
├── inference/          # Inference engines
├── training/           # Training implementations
└── utils/              # Utility functions
```

### 🔧 Core Components

#### `core/`
Contains abstract base classes that define common interfaces for all training and inference implementations. Provides the foundation that other modules inherit from to ensure consistency across different approaches.

### 📊 Data Processing

#### `data/`
Handles data loading, preprocessing, and formatting for different dataset types. Supports multiple formats including Alpaca instruction datasets, mathematical problem datasets, and generic CSV data. Includes tokenization utilities for model training.

### 🧪 Evaluation Tools

#### `evaluation/`
Provides model evaluation capabilities using LLM-as-a-Judge methodology with Gemini API integration. Compares model outputs and provides structured scoring for mathematical problem-solving tasks.

**Gemini LLM Judge Key Features:**
- **Synchronous Evaluation**: Standard evaluation using `LLMGeminiJudgeEvaluator`
- **Asynchronous Evaluation**: High-performance async evaluation using `LLMGeminiJudgeEvaluatorAsync`
- **Batch Processing**: Optimized batch processing for both CPU-bound (model generation) and I/O-bound (API calls) operations
- **Memory Management**: Intelligent memory cleanup between evaluation phases
- **Rate Limiting**: Built-in rate limiting to handle API constraints
- **Intermediate Saving**: Automatic saving of evaluation results with timestamps
- **Performance Optimization**: Configurable worker pools and batch sizes for optimal throughput

### 🚀 Inference Engines

#### `inference/`
Multiple inference implementations optimized for different use cases:
- **Standard**: Compatible transformers-based inference with PEFT support
- **Unsloth**: Optimized for 2x faster inference and memory efficiency  
- **vLLM**: High-throughput serving with GPU optimization and LoRA support

### 🎓 Training Implementations

#### `training/`
Specialized trainers for different optimization approaches:
- **Unsloth**: Fastest training with lowest memory usage
- **LoRA**: Standard PEFT implementation with broad compatibility
- **LLMCompressor**: Quantized training with model compression support

### 🛠️ Utilities

#### `utils/`
Common utility functions for memory management, CUDA operations, and model diagnostics. Includes tools for monitoring GPU usage, clearing cache, and getting model statistics.

### 📜 Scripts

#### `Scripts/`
End-to-end automation scripts for complete training and evaluation workflows.

**script1.py - Complete Pipeline Test Script:**
- **Purpose**: Simplified end-to-end testing of the entire SLM pipeline
- **Features**: 
  - Single-parameter configuration for quick testing
  - Complete workflow: Training → Standard Inference → vLLM Inference → Async Evaluation
  - Memory optimization with cleanup between phases
  - Comprehensive logging and error handling
  - JSON output with detailed results
  - Async evaluation for all inference methods

**Pipeline Stages:**
1. **Model Training**: LoRA fine-tuning with configurable parameters
2. **Standard Inference**: Traditional transformers-based inference
3. **vLLM Inference**: High-performance inference engine
4. **Async Evaluation**: Parallel evaluation using Gemini API with optimized batching
5. **Results Compilation**: Structured JSON output with comparison metrics

**Usage:**
```bash
python Scripts/script1.py
```
### 🎯 Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new trainers/inference engines  
3. **Consistency**: Common interfaces across implementations
4. **Performance**: Optimized implementations for different use cases
5. **Flexibility**: Configuration-driven behavior

### 📋 Usage Guidelines

- **Memory Limited**: Use Unsloth components for training and inference
- **Speed Critical**: Use vLLM for inference, Unsloth for training  
- **Model Compression**: Use LLMCompressor trainer for quantized models
- **Standard Compatibility**: Use standard implementations for broad compatibility
- **Large-Scale Evaluation**: Use async evaluation components for efficient batch processing

### ⚙️ Example Configurations

#### Training Configuration
```python
config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_new_tokens": 128,
    "device_map": "auto",
    "seed": 108
}
```

#### Inference Configuration
```python
config = {
    "base_model": "Qwen/Qwen2-0.5B",
    "dtype": None,
    "device_map": "auto",
    "max_new_tokens": 128,
    "temperature": 0.5,
    "gpu_memory_utilization": 0.3,
    "do_sample": False
}
```

#### Async Evaluation Configuration
```python
async_config = {
    "cpu_workers": 4,        # CPU-bound operations (model generation)
    "io_workers": 10,        # I/O-bound operations (API calls)
    "cpu_batch_size": 2,     # Batch size for GPU operations
    "io_batch_size": 5       # Batch size for API operations
}
```

---

This modular architecture enables flexible experimentation with different training techniques and inference optimizations while maintaining code reusability and clean interfaces.

## References
- **[Unsloth](https://unsloth.ai/)** - Fast and memory-efficient fine-tuning
- **[LLMCompressor](https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm#enabling_activation_quantization_in_vllm)** - Model compression for faster inference
- **[vLLM](https://docs.vllm.ai/en/latest/)** - High-performance LLM serving framework
- **[My Notebook](https://notebooklm.google.com/notebook/eee2c93a-12a8-4dba-9311-a76b464c58ac)** - Learnings and experiment results

## Contributing
Feel free to raise an issue or submit a pull request if you find any mistakes or have suggestions for improvement. Your contributions are welcome and appreciated!

---

Happy Coding!