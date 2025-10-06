from self_consistency_code_generation import SelfConsistencyCodeGeneration, TokenUsage
import os
from humaneval_loader import HumanEvalLoader
from BigCodeLoader import BigCodeLoader
from MBPPLoader import MBPPLoader
from concurrent.futures import ThreadPoolExecutor
from evaluator import CodeEval
from self_consistency_code_generation import CodeEmbedder
import json
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def worker(problem, api_key, model_name, embedder):
    sc_generator = SelfConsistencyCodeGeneration(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        embedder=embedder
    )
    result = sc_generator.self_consistency_generate(
        problem=problem,
        num_samples=5
    )
    # print(f'Problem: {problem}\nResult: {result}\n')
    if result["success"]:
        print(f"\nConfidence: {result['confidence']:.2f}")
        return result["final_code"], sc_generator.token_usage
        # print(f"Solutions generated: {result['num_solutions_generated']}")
        # print(f"Vote distribution: {result['vote_distribution']}")
    else:
        print(f"Generation failed: {result['error']}")
        return '', sc_generator.token_usage

def main():
    """
    Example usage of Self-Consistency Code Generation with o3-mini
    """
    # Initialize with your OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    total_token_usage = TokenUsage()
    api_key = os.getenv('openai_key')
    model_name = 'o3-mini-2025-01-31'
    dataset_name = 'BigCode'
    if dataset_name == 'HumanEval':
        dataset = HumanEvalLoader().get_human_eval()
        dataset = [hh['prompt'] for hh in dataset['test']]
        final_test_cases = HumanEvalLoader().get_final_test_cases()
    elif dataset_name == 'BigCode':
        loader = BigCodeLoader(hard=1)
        dataset =loader.get_prompts()
        final_test_cases = loader.get_tests()
        bigcode_dis = loader.get_ids()
    elif dataset_name == 'MBPP':
        loader = MBPPLoader()
        dataset = loader.get_prompts()
        final_test_cases = loader.get_tests()
    else:
        raise ValueError('Dataset name must be one of HumanEval or BigCode')
    print(dataset[0])
    print(len(dataset))
    print(len(final_test_cases))
    # dataset = dataset[:5]
    embedder = CodeEmbedder("microsoft/codebert-base")
    # Initialize the generator
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(worker, prob, api_key, model_name, embedder)
            for prob in dataset
        ]
        results = [f.result() for f in futures]

    final_codes = [res[0] for res in results]
    token_usages = [res[1] for res in results]
    fillings = [[code] for code in final_codes]

    if dataset_name == 'BigCode':
        out_dict = []
        for index, item in enumerate(dataset):
            out_dict.append({
                'prompt': item,
                'solution': fillings[index][0],
                'test_cases': final_test_cases[index],
                'name': bigcode_dis[index],
                'task_id': bigcode_dis[index],
            })
        file_dir = f'BigCode_self_consistency.jsonl'
        with open(file_dir, "w") as file:
            for item in out_dict:
                file.write(json.dumps(item) + "\n")
    else:
        pass_at_k, eval_results = CodeEval()._compute(references=final_test_cases, predictions=fillings, k=[1])
        print(pass_at_k)
        print(eval_results)
    total_token_usage = sum(token_usages, TokenUsage())  # Assuming TokenUsage supports addition
    print(f"Total token usage: {total_token_usage}")

if __name__ == '__main__':
    main()


