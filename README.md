# GPT-2 Fine-Tuning Project

## Overview
A complete implementation for fine-tuning GPT-2 (355M parameter version) on custom instruction-following tasks. The model is trained to understand structured prompts and generate appropriate responses.

## Key Features

### Model Architecture
- 🧬 Custom GPT-2 implementation with:
  - Multi-head attention layers
  - Position-wise feedforward networks
  - Layer normalization
  - 24 transformer blocks
- 🎛 Model Config:
  ```python
  BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "drop_rate": 0.0
  }


### Dataset Handling
- 📂 1,100 instruction examples (85% train, 10% test, 5% validation)
- 📝 Example format:
  ```json
  {
    "instruction": "Convert to passive voice",
    "input": "The chef cooks the meal",
    "output": "The meal is cooked by the chef"
  }
  ```
- 🛠 Custom preprocessing:
  ```python
  def format_input(entry):
    base = "Below is an instruction...\\n### Instruction:\\n{instruction}"
    return base + (f"\\n### Input:\\n{entry['input']}" if entry["input"] else "")
  ```


### Core Dependencies
```python
torch==2.0.1
tensorflow==2.15.0
tqdm==4.66.1
tiktoken==0.9.0
numpy==1.23.5
```

### Hardware
- NVIDIA GPU (T4 or better recommended)
- 16GB+ VRAM
- Google Drive for model storage


## Training Process

### Hyperparameters
| Parameter        | Value   |
|------------------|---------|
| Batch Size       | 8       |
| Learning Rate    | 5e-5    |
| Context Length   | 1024    |
| Warmup Steps     | 100     |
| Weight Decay     | 0.1     |
| Epochs           | 2       |

### Loss Curves
![output](https://github.com/user-attachments/assets/341ecd7e-9c2a-4f66-88bb-2bd7d45ca315)



## Results

### Quantitative Metrics
| Metric           | Value   |
|------------------|---------|
| Train Loss       | 0.68    |
| Val Loss         | 0.72    |
| Inference Speed  | 12t/s   |
| Accuracy         | 78.4%   |

### Qualitative Examples
| Instruction                          | Correct Answer                 | Model Output                     |
|--------------------------------------|---------------------------------|----------------------------------|
| Rewrite using simile: Fast car       | "as fast as lightning"         | "as fast as a bullet"            |
| Author of Pride and Prejudice        | "Jane Austen"                  | "The author... is Jane Austen"   |
| Periodic symbol for chlorine         | "Cl"                           | "C"                              |


## License
MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments
- Original GPT-2 paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Dataset adapted from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
