import random
import re
import reasoning_gym
from reasoning_gym import get_score_answer_fn
from functools import partial

brainstorm_sentences = [
    "\nProvide the final answer on last line after brainstorming.",
    "\nThink through the problem step by step, then present the final answer on the last line.",
    "\nDo your reasoning internally and only state the final conclusion at the end.",
    "\nAnalyze all possibilities first and write the final answer on the last line.",
    "\nBrainstorm thoroughly before giving the final answer as the last line.",
    "\nWork out the logic carefully, then provide the final answer at the end.",
    "\nConsider intermediate steps silently and output only the final answer on the last line.",
    "\nReason about the problem in detail, but show the final answer only at the end.",
    "\nEvaluate the problem step by step and conclude with the final answer on the last line.",
    "\nThink carefully through all steps, then write the final answer on the last line.",
    "\nPerform detailed reasoning first and place the final answer at the very end.",
    "\nInternally deliberate before responding, and give the final answer on the last line.",
    "\nComplete all analysis before presenting the final answer as the last line."
    "",
    ""
]

def number_sorting_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    last_line = last_line[last_line.find('['):]
    return score_fn(last_line, entry)

def number_sorting(tokenizer, size):
    dataset = reasoning_gym.create_dataset(
        name="number_sorting",   # task name
        min_numbers = 3,
        max_numbers = 10,
        min_decimals = 0,
        max_decimals = 1,
        min_value = -20,
        max_value = 20,
        seed = 42,
        size = size,
        # num_fewshot=1,
        # fewshot_as_multiturn=1
    )
    score_fn = get_score_answer_fn("number_sorting")
    instruction = ' Reply only with a name.'
    dataset_list = []

    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'].rstrip(instruction) + random.choice(brainstorm_sentences) + f'\n{instruction}'}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(number_sorting_parser, entry=data, score_fn=score_fn)
        })

    return dataset_list


def needle_haystack_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    name = last_line.strip().split()[-1].lower()
    return score_fn(name, entry)


def needle_haystack(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="needle_haystack",   # task name
        min_num_statements = 2,
        max_num_statements = 100,
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("needle_haystack")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(needle_haystack_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def syllogism_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    name = last_line.strip().split()[-1].capitalize()
    return score_fn(name, entry)

def syllogism(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="syllogism",   # task name
        allow_all = True,
        allow_no = True,
        allow_some = True,
        allow_some_not = True,
        invalid_ratio = 0.3,
        inversion_probability = 0.3,
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("syllogism")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(syllogism_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def alice_in_wonderland_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'\d+', last_line)
    if digits:
        return score_fn(digits[-1], entry)
    return 0


def alice_in_wonderland(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="aiw",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("aiw")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(alice_in_wonderland_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def family_relationships_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    # name = last_line.strip().split()[-1].lower()
    return int(entry['answer'].lower() in last_line.lower().split())
    # return score_fn(name, entry)

def family_relationships(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="family_relationships",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("family_relationships")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(family_relationships_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def gsm_symbolic_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'\d+', last_line)
    if digits:
        return score_fn(digits[-1], entry)
    return 0


def gsm_symbolic(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="gsm_symbolic",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("gsm_symbolic")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(gsm_symbolic_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def list_functions_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    last_line = last_line[last_line.find('['):]
    return score_fn(last_line, entry)


def list_functions(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="list_functions",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("list_functions")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(list_functions_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list



def codeio_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    return score_fn(last_line, entry)


def codeio(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="codeio",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("codeio")

    dataset_list = []
    for data in dataset:
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question'] + random.choice(brainstorm_sentences)}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(codeio_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


if __name__ == '__main__':
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")

    ds = alice_in_wonderland(tokenizer=tokenizer)

    print(ds[-1]['prompt'])
    answer = "THINKING...\nFinal Answer: " + ds[-1]['answer']
    print(answer)
    print(ds[-1]['scorer'](answer))