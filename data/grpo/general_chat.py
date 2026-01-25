# See: https://huggingface.co/datasets/allenai/Dolci-RL-Zero-General-7B
from functools import partial
from datasets import load_dataset
from .sandbox import DockerSandbox
from .verifiers import get_llm_response, response_judge

def general_chat_scorer(llm_gen, llm_judge, question, ground):
    if len(llm_gen.strip()) <= 256: return 0
    score = response_judge(question=question, response=llm_gen, ref_answer=ground, n_tokens=512, strict_level=2)[-1]
    score = (score * 3) / 2
    return min(score, 1)
    # return score

def general_chat_ds(tokenizer, prompt_token_len):
    dataset = load_dataset("allenai/Dolci-RL-Zero-General-7B")['train']
    dataset = dataset.map(lambda x: {'user_question': x['prompt'].strip().lstrip('user:').strip()})
    dataset = dataset.map(lambda x: {
        'prompt': tokenizer.apply_chat_template(
            [{'role': 'user', 'content': x['user_question']}],
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
        ),
        'ground_truth': x['ground_truth'][0]
    })
    dataset = dataset.remove_columns(['custom_id'])
    dataset = [d for d in dataset]
    for i in range(len(dataset)):
        dataset[i]['scorer'] = partial(
            general_chat_scorer, 
            question=dataset[i]['user_question'],
            ground=dataset[i]['ground_truth']
        )
    dataset = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset))
    return dataset


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")
    dataset = general_chat_ds(tokenizer, 256)
    print(dataset[0])
    scorer = dataset[0]['scorer']
    print(scorer(dataset[0]['ground_truth'], True))
