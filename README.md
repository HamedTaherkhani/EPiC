# Evolutionary Prompt Engineering for Cost-Effective Code Generation with Large Language Models

Welcome to EPiC, a framework designed to **cost-effectively** generate high-quality code by iteratively improving prompts through a **lightweight evolutionary algorithm**. EPiC (Evolutionary **P**rompt **En**gineering for **C**ode) refines an original prompt to produce increasingly better code solutions, all while minimizing the number of calls (and hence cost) to large language models (LLMs). 

This repository contains:

1. **`run_experiments.py`**  
   Main script to run various experiment configurations with different LLMs and evolutionary settings.

2. **`testcase_generator` directory** (and associated scripts like `testcase_generator.py`)  
   Scripts to generate test cases (HumanEval, MBPP, BigCodeBench, etc.) using different models and approaches.

3. **`requirements.txt`**  
   Python dependencies needed to reproduce the experiments.

---

## Project Overview

**EPiC** is an approach that uses an evolutionary algorithm to **refine prompts** for code generation tasks. While many agent-based or iterative code-generation methods make extensive LLM calls (thus driving up costs), EPiC aims to achieve:

- **High-quality code** (measured via pass rates on test suites).
- **Cost-effectiveness** (minimizing token usage and total calls to LLMs).

### Key Highlights
- **Lightweight Evolutionary Algorithm**: EPiC starts with an initial prompt, evaluates the generated code, and if incorrect, mutates the prompt in a minimal-cost manner until a valid solution is found.
- **Flexible**: Can be adapted to multiple LLMs (e.g., GPT-4, MagicCoder, Claude 3.7, DeepSeek, etc.) and various code datasets (HumanEval+, MBPP+, BigCodeBench, etc.).
- **Local or LLM-based Mutation**: Supports two types of mutation operators:
  1. **`sim_words_as_mutator`** using local word embeddings (NLTK + Gensim) to cheaply vary prompt text.
  2. **`llm_as_mutator`** using the LLM to mutate the prompt for more sophisticated changes (at higher cost).

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUserName/EPiC.git
   cd EPiC
   ```
```bash
# using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
```bash
# For running bigcodebench you need a separate env
python3.10 env .bigcode_venv
source .bigcode_venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_bigcode.txt

```

## Usage


### Setup
- get openAI key from https://platform.openai.com/api-keys (for o3-mini)
- get Firework key from https://fireworks.ai/account/api-keys (for deepseek-v3)
- get Antropic key from https://console.anthropic.com/settings/keys (for claude 3.7 Sonnet)
### Environment Variables
EPiC uses environment variables for flexible configuration. You can create a `.env` file or set these environment variables manually.

For example, your `.env` might look like:
```
experiment=6
human_eval_instances=[] ## A list of selected instance IDs (used in some experiments). Leave it empty as default.
openai_key=your_openai_api_key
anthropic_key=your_anthropic_api_key
fireworks_key=your_fireworks_api_key
```

### Running Experiments with `run_experiments.py`

The script **`run_experiments.py`** is the central entry point. It orchestrates multiple types of experiments, each identified by an integer ID in the `experiments` dictionary:

```python
experiments = {
    1: 'genetic-magiccoder-llama2-70b',
    2: 'genetic-codellama-llama2-70b',
    3: 'genetic-magiccoder-gensim',
    ...
    22: 'genetic-sonnet3.7-mbpp',
}
```

**Steps to run**:

1. Set `experiment` in your `.env` or environment.
2. Run:
   ```bash
   python run_experiments.py
   ```
3. The script automatically chooses the appropriate `Runner` or `Experiments` class to execute the genetic (evolutionary) prompt engineering procedure.

---

### Generating Test Cases

We employ LLMs to generate test cases for benchmarks (like HumanEval or MBPP) in a fully automated manner:

- **`testcase_generator.py`** (entry script)  
  - **`generate_for_humaneval(model_name)`**: Generates test cases for the HumanEval dataset.  
  - **`generate_for_mbpp(model_name)`**: Generates test cases for the MBPP dataset.  

**Usage**:
1. Update `model_name` in `testcase_generator.py`.
2. Run:
   ```bash
   python testcase_generator.py
   ```
3. The generated test cases are stored in `testcases/`.

---

## Directory Structure

A brief overview of key files/folders:

```
EPiC/
  ├─ run_experiments.py       # Main script for running genetic prompt engineering experiments
  ├─ testcase_generator.py     # Test-case generation script
  ├─ BigCodeLoader.py          # Utility to handle BigCodeBench dataset
  ├─ MBPPLoader.py             # Utility to handle MBPP dataset
  ├─ humaneval_loader.py       # Utility to load HumanEval dataset
  ├─ ...
  ├─ TestcaseGenerator/        # Directory containing various test generation utilities
  │   ├─ generators/           # Different generator factories
  │   └─ ...
  ├─ .env.example             # Example environment file
  ├─ requirements.txt          # All Python dependencies
  └─ paper/ or paper text      # The draft paper describing EPiC
```

---

## Key Components

### 1. The EPiC Algorithm

In **`run_experiments.py`**, each experiment uses **EPiC**:

1. **Initial Evaluation (IE)**:
   - Prompt the LLM to generate code and test cases.
   - Evaluate the code with the newly generated tests.  
   - If the code fails tests, proceed to evolutionary prompt engineering.

2. **Evolutionary Prompt Engineering (EPE)**:
   - Create a **population** of mutated prompts (the first generation).
   - For each prompt:
     - Ask the LLM for code.
     - Evaluate against tests → get a **fitness** score.
   - **Select** top prompts based on fitness and **mutate** them to form the next generation.
   - Repeat until a correct (fully passing) solution emerges or iteration-limit reached.

### 2. Mutation Approaches

Two main mutation modes:

1. **`llm_as_mutator`**  
   - Uses an LLM to rewrite the prompt.  
   - Potentially more powerful but higher API cost.

2. **`sim_words_as_mutator`**  
   - Uses NLP libraries (NLTK + Gensim) to find synonyms or similar words.  
   - Low cost but still effective at nudging the LLM to generate alternative code.

### 3. Datasets & Benchmarks

We focus on three main benchmarks:

1. **HumanEval+**: Extended HumanEval with more thorough test cases.  
2. **MBPP+**: Extended MBPP with additional tests and broader coverage.  
3. **BigCodeBench**: Large dataset spanning more complex function calls and instructions.

Each dataset is loaded through a corresponding loader (e.g., `HumanEvalLoader`, `MBPPLoader`, `BigCodeLoader`).
