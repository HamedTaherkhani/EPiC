# EPiC: Cost-effective Search-based Prompt Engineering of LLMs for Code Generation

## Introduction
This project presents EPiC, a cost-effective approach leveraging a lightweight evolutionary algorithm to evolve prompts for better code generation using Large Language Models (LLMs). The framework outperforms state-of-the-art methods in terms of cost-effectiveness and accuracy.

## Project Structure
The main files in this project are:

- `run_experiments.py`: Main script to execute various experiments.
- `testcase_generator.py`: Generates test cases for MBPP and HumanEval datasets.

## Installation
To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Running Experiments
Experiments can be run using the `run_experiments.py` script. This script uses several modules and configurations to perform different experiments. The environment variables and configurations must be set in a `.env` file.

### Environment Variables
Create a `.env` file in the root directory of the project and set the following variables:

```env
experiment=<experiment_id>
openai_key=<openai_key>
openai_model=<openai_model>
```

### Available Experiments
The available experiments are defined in the `run_experiments.py` script. Here are the available experiment IDs and their corresponding descriptions:

1. `genetic-magiccoder-llama2-70b`
2. `genetic-codellama-llama2-70b`
3. `genetic-magiccoder-gensim`
4. `genetic-codellama-gensim`
5. `genetic-magicoder-llama2-7b`
6. `genetic-gpt4-gensim`
7. `genetic-gpt4-gensim-ten_population`
8. `genetic-gpt4-gensim-v2`
9. `genetic-gpt4-gensim-10-times`
10. `genetic-gpt4-gensim-mbpp`
11. `genetic-gpt4-gpt4`
12. `genetic-gpt4-gensim-mbpp-10-times`

### Running a Specific Experiment
To run a specific experiment, set the `experiment` variable in the `.env` file to the desired experiment ID. Then run:

```bash
python run_experiments.py
```

### Example `.env` File
```env
experiment=9
openai_model=gpt-4o
openai_key=your_key
```

## Generating Test Cases
To generate test cases, use the `testcase_generator.py` script. This script can generate test cases for both the MBPP and HumanEval datasets.

## Citation
If you use this code, please cite our paper:

```bibtex
@article{taherkhani2025epic,
  title={EPiC: Cost-effective Search-based Prompt Engineering of LLMs for Code Generation},
  author={Taherkhani, Hamed and Sepindband, Melika and Pham, Hung Viet and Wang, Song and Hemmati, Hadi},
  journal={arXiv preprint arXiv:2308.12950},
  year={2025}
}
```
