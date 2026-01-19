# See: https://huggingface.co/datasets/allenai/Dolci-RL-Zero-General-7B
from functools import partial
from datasets import load_dataset
from .sandbox import DockerSandbox
from .verifiers import get_llm_response, response_judge


def general_ds(tokenizer, prompt_token_len):
    dataset = load_dataset("Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs")['train']
    dataset = dataset.map(lambda x: {'eval_funcs': list(set(x['eval_funcs']))})
    dataset = dataset.map(lambda x: {
        'prompt': tokenizer.apply_chat_template(
            x['messages'][:-1],
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
    )})
    dataset = dataset.remove_columns(['system', 'tools', 'conversation_id'])
    dataset = [d for d in dataset]
    for i in range(len(dataset)):
        dataset[i]['scorer'] = partial(scorer, eval_func=dataset[i]['eval_funcs'], question=dataset[i]['messages'][-1]['content'])
    dataset = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset))
    return dataset
