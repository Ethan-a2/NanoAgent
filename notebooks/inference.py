#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys

sys.path.append("../")
from utils.tokenizer import TOOL_TEMPLATE_PY


def json_toolcall_to_python(tool_calls, markdown_format=True):
    """Convert a JSON tool-call into a Python-style function call string."""

    def format_value(value):
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return "[" + ", ".join(format_value(v) for v in value) + "]"
        elif isinstance(value, dict):
            return (
                "{"
                + ", ".join(
                    f"{format_value(k)}: {format_value(v)}" for k, v in value.items()
                )
                + "}"
            )
        return str(value)

    if isinstance(tool_calls, str):
        tool_calls = json.loads(tool_calls)
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    returns = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {})
        if not name:
            continue
        args_str = ", ".join(f"{k}={format_value(v)}" for k, v in args.items())
        returns.append(f"{name}({args_str})")
    if markdown_format:
        return "```python\n" + "\n".join(returns) + "\n```"
    return returns


def json_tooldef_to_python(tools, indent=None):
    """Convert JSON tool descriptions into Python function definitions."""

    def map_type(prop):
        t = prop.get("type", "any")
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        return type_map.get(t, "Any")

    funcs = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        args_list = []
        for pname, pval in params.items():
            ptype = map_type(pval)
            default = "" if pname in required else " = None"
            args_list.append(f"{pname}: {ptype}{default}")

        args_str = ", ".join(args_list)
        func_def = f'def {name}({args_str}):\n    """{desc}"""\n    ...'
        funcs.append(func_def)

    return "\n\n".join(funcs)


# In[ ]:


# model_name = "quwsarohi/NanoAgent-135M"
# model_name = "/Users/ohi/Documents/GitHub/NanoAgent/weights/NanoAgent-135M-8bit"
# model_name = "../weights/SmolLM2-135M-Instruct-agentic-dft"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model_name = "../weights/NanoAgent-135M-nemotron-dft"
# model_name = "../weights/NanoAgent-135M-grpo-math"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True, dtype=torch.bfloat16
)
tokenizer.pad_token_id = 3


def inference(messages, max_new_tokens=256, temperature=0.0, min_p=0.15, **kwargs):
    if isinstance(messages, list):
        continue_final_message = messages[-1]["role"] == "assistant"
        messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
    inputs = tokenizer.encode(messages, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        min_p=min_p if temperature > 0 else None,
        temperature=temperature if temperature > 0 else None,
        **kwargs,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)


# In[12]:


TOOLS = [
    {
        "name": "web_search",
        "description": "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "visit_webpage",
        "description": "Fetches and displays the textual content of a webpage (converted to Markdown) from a given URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to retrieve.",
                }
            },
            "required": ["url"],
        },
    },
    # {
    #     "name": "final_answer",
    #     "description": "Provides a final answer to the given problem.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "answer": {
    #                 "type": "string",
    #                 "description": "The final answer to the problem",
    #             }
    #         },
    #         "required": ["answer"],
    #     },
    # },
]

# import random
# random.shuffle(TOOLS)
# TOOLS = json_tooldef_to_python(TOOLS, indent=2)
# # print(TOOLS)

TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible functions/tools inside <tools></tools> tags. 
Based on question, you may need to make one or more function/tool calls to answer user.

You have access to the following tools/functions:
<tools>{tools}</tools>

For each function call, return a JSON list object with function name and arguments within <tool_call></tool_call> tags."""
TOOL_TEMPLATE = TOOL_TEMPLATE.format(tools=TOOLS)

# TOOL_TEMPLATE = TOOL_TEMPLATE_PY().format(tools=TOOLS)

# TOOL_TEMPLATE += "\nPrefer doing a web_search before answering user query."
# print(TOOL_TEMPLATE)

messages = [
    # {"role": "system", "content": TOOL_TEMPLATE},
    {
        "role": "user",
        "content": "Define machine learning.\nAfter that, List 3 domains where ML can be applied in bullet points.",
    },  # Return your answer in JSON as the following format:\n```json\n{'concept': 'your_concept', 'definition': ...}\n```"},
    # {"role": "user", "content": "Who is current the president of Bangladesh?"},
    # {"role": "user", "content": "Write an email to my manager asking for parental leave extension."},
    # {"role": "assistant", "content": "```python\n"}
    # {"role": "assistant", "content": "The "}
    # {"role": "user", "content": "Hi"},
]

print("-" * 30)
print(
    inference(
        messages, max_new_tokens=384, min_p=0.2, temperature=0.1, repetition_penalty=1.1
    )
)  # min_p=0.2, temperature=0.1, repetition_penalty=1.05


# In[4]:


# sys.exit()


# In[ ]:


tokenizer.encode("<|im_end|>")


# In[ ]:


tokenizer.chat_template = """{% for m in messages %}<|begin_of_text|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<think>
{% endif %}"""

print(tokenizer.pad_token_id, tokenizer.eos_token_id)

messages = [
    # {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "What is 4+9?"},
    # {"role": "assistant", "content": "```json\n"}
]

prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

print(prompt)
print("-+" * 20)

print(inference(prompt, max_new_tokens=256, temperature=0.3))


# In[ ]:


print(tokenizer.pad_token_id, tokenizer.eos_token_id)


# In[ ]:


# messages = [{"role": "user", "content": "In a communication system, data is sent using orthogonal Latin squares. The transmission is organized into 4 time slots. Each slot carries 3 distinct symbols. During the transmission, an additional 4 symbols are sent from a separate source. What is the total number of symbols sent during this transmission?"}]
messages = [{"role": "user", "content": "Define Machine Learning in 3 bullet points."}]
# messages = [
#     # {"role": "system", "content": "You are a helpful AI assistant. You must think step-by-step inside <think> </think> tags before answering."},

# #     {"role": "user", "content": """In this task you are given a sentence. You must judge whether subject of the main clause is singular or plural. Label the instances as "Singular" or "Plural" based on your judgment.
# # Q: Kids and teachers swarmed the area like a bunk of flies to a Bar-B-Q.
# # A:  Do not include keyword on in the response. Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."""}
# ]

print()

for _ in range(3):
    print(
        inference(
            messages,
            max_new_tokens=128,
            temperature=0.3,
            top_k=30,
            repetition_penalty=1.05,
            min_p=None,
        )
    )
    print("---")


# In[ ]:


# messages = [{"role": "user", "content": "In a communication system, data is sent using orthogonal Latin squares. The transmission is organized into 4 time slots. Each slot carries 3 distinct symbols. During the transmission, an additional 4 symbols are sent from a separate source. What is the total number of symbols sent during this transmission?"}]
messages = [
    {
        "role": "user",
        "content": "What is machine learning? Write your answer in bullet points.",
    }
]
# messages = [{"role": "user", "content": "What is 8+9="}]
print(inference(messages, max_new_tokens=128, temperature=0.3))


# In[ ]:


# # messages = [{"role": "user", "content": "Hi!"}]
# input_text = \
# """<|im_start|>user
# Hi!<|im_end|>
# <|im_start|>assistant
# """
# print(inference(input_text, max_new_tokens=1024))


# In[ ]:


TOOLS = [
    {
        "name": "web_search",
        "description": "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "visit_webpage",
        "description": "Fetches and displays the textual content of a webpage (converted to Markdown) from a given URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to retrieve.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "final_answer",
        "description": "Provides a final answer to the given problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the problem",
                }
            },
            "required": ["answer"],
        },
    },
]

TOOLS = json_tooldef_to_python(TOOLS, indent=2)
# print(TOOLS)

# TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible functions/tools inside <tools></tools> tags.
# Based on question, you may need to make one or more function/tool calls to answer user.

# You have access to the following tools/functions:
# <tools>{tools}</tools>

# For each function call, return a JSON list object with function name and arguments within <tool_call></tool_call> tags."""

TOOL_TEMPLATE = TOOL_TEMPLATE_PY().format(tools=TOOLS)
# print(TOOL_TEMPLATE)

messages = [
    {"role": "system", "content": TOOL_TEMPLATE},
    # {"role": "user", "content": "What capabilities do you have?"},
    {
        "role": "user",
        "content": "Who is the current president of USA as of 2012? Do a web search and then give a final answer",
    },
    # {"role": "assistant", "content": "```python\n"}
]

print("-" * 30)
print(inference(messages, min_p=0.2))


# In[ ]:


from ddgs import DDGS


def web_search(query: str, max_results: int = 2) -> str:
    """
    Performs a DuckDuckGo web search and returns formatted markdown output.

    Args:
        query (str): Search query string.
        max_results (int): Number of results to return.

    Returns:
        str: Markdown formatted search results.
    """
    results = DDGS().text(query, max_results=max_results)

    output_lines = ["## Search Results"]

    for i, item in enumerate(results):
        title = item.get("title", "No Title")
        href = item.get("href", "")
        snippet = item.get("body", "")
        date = item.get("date", "Unknown date")

        output_lines.append(
            f"{i}. [{title}]({href})\nDate published: {date}\n\n{snippet}\n"
        )

    return "\n".join(output_lines)


# In[ ]:


import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def visit_webpage(url: str, main_selector: str = None) -> str:
    """
    Fetch a URL, extract HTML content, convert to Markdown, and return it.

    Args:
        url (str): The webpage URL.
        main_selector (str): Optional CSS selector to extract specific content
                             (e.g. "#content", ".article", "main", etc.)

    Returns:
        str: Markdown text extracted from the page.
    """

    # Fetch page
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main content if selector provided
    if main_selector:
        element = soup.select_one(main_selector)
        if not element:
            raise ValueError(f"No element found for selector: {main_selector}")
        html_content = str(element)
    else:
        # Fallback: use <body>
        html_content = str(soup.body or soup)

    # Convert HTML → Markdown
    markdown_text = md(html_content, heading_style="ATX")

    return markdown_text


# In[ ]:


def wiki_search(inp: str):
    import wikipedia

    result = None
    try:
        result = wikipedia.summary(inp)
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options
        results = []
        for option in options:
            results.append(f"# Topic: {option}\n{wikipedia.summary(option)}")
        result = "\n".join(results)
    return json.dumps({"search_result": result.strip()})


def extract_toolcall(input_str):
    import re

    calls = re.findall(r"<tool_call>(.*?)</tool_call>", input_str, re.DOTALL)
    if calls:
        calls = calls[-1].strip()
        try:
            return json.loads(calls)
        except:
            return []
    return []


# messages = [
#     {
#         "role": "system",
#         "content": TOOL_TEMPLATE.format(tools=json.dumps(TOOLS))
#         #+ "Think step by step before answering.",
#     },
#     {
#         "role": "user",
#         # "content": "Hey, what is the scientific name of yellow toadflax?",
#         "content": "When Dr Yunus he became elected as president??"
#     },
#     # {"role": "assistant", "content": "<think>"}
# ]

# print("User:", messages[-1]["content"])
# llm_tool_call = inference(messages, max_new_tokens=512)
# print("LLM:", llm_tool_call)

# search_query = extract_toolcall(llm_tool_call)[0]["arguments"]["query"]
# # search_result = wiki_search(search_query)
# search_result = web_search(search_query, max_results=4)
# tool_reply = f"<tool_response>{search_result}</tool_response>"

# print("--- Web Result ---")
# print(search_result)
# print("-------------------\n")

# messages += [
#     {"role": "assistant", "content": llm_tool_call},
#     {"role": "user", "content": tool_reply},
# ]

# ret = inference(messages, max_new_tokens=512)
# print("LLM:", ret)


# In[ ]:


def iterate_tools(user_question):
    messages = [
        {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(TOOLS))},
        {"role": "user", "content": user_question},
    ]
    tool_call_lead = [{"role": "assistant", "content": "<tool_call>["}]
    print(tokenizer.apply_chat_template(messages, tokenize=False))

    for itr in range(4):
        # llm_gen = "<tool_call>[" + inference(messages + tool_call_lead, max_new_tokens=512)
        llm_gen = inference(messages, max_new_tokens=512, temperature=0.3, min_p=0.1)
        messages.append({"role": "assistant", "content": llm_gen})
        print("LLM:", llm_gen)

        try:
            # if True:
            gen_tool = extract_toolcall(llm_gen)[0]
            if "web_search" == gen_tool["name"]:
                result = web_search(gen_tool["arguments"]["query"], max_results=5)
            elif "final_answer" in gen_tool["name"]:
                result = gen_tool["arguments"]["answer"]
                return result
            elif "visit_webpage" in gen_tool["name"]:
                result = visit_webpage(gen_tool["arguments"]["url"])

            tool_reply = f"<tool_response>{result}</tool_response>"
            messages.append({"role": "user", "content": tool_reply})
            print("## Tool reply:\n", tool_reply)
        except:
            pass

    return llm_gen


iterate_tools("Hey, what is the scientific name of yellow toadflax?")
# iterate_tools("Hey, what is the weather in ottawa?")
# iterate_tools("What ingerients are needed to make a cake?")


# In[ ]:


# # Example usage:
# if __name__ == "__main__":
#     url = "https://www.bbc.com/news/articles/clyg7we8xvno"
#     markdown = visit_webpage(url, main_selector="body")
#     print(markdown)


# In[ ]:


import re


def add_missing_periods(text: str) -> str:
    """
    Inserts periods between sentences when they are missing.
    Does not split on acronyms (e.g., XYZ).
    """
    # Add a period before a capitalized word that follows a lowercase letter
    # Example: "sentence This" -> "sentence. This"
    pattern = r"(?<=[a-z])\s+(?=[A-Z][a-z])"
    return re.sub(pattern, ". ", text)


s = "This is sentence with acronym XYZ"
print(add_missing_periods(s))

s = "This is sentence This is another sentence"
print(add_missing_periods(s))


# In[ ]:


import re

s = r"The answer is \boxed{12/2}"
print(s)


def extract_boxed(inp):
    match = re.search(r"\\boxed\{(.*?)\}", inp)
    if match:
        return match.group(1)


print(extract_boxed(s))


# In[ ]:


ans = """In the market, there are 8 pink balls, 10 yellow balls, and 24 green balls. Each ball costs $15. So the total cost of the balls would be 8 * $15 = $110.

There are 8 pink balls, so the total cost is 8 * $15 = $110.

There are 10 yellow balls, so the total cost is 10 * $15 = $150.

There are 24 green balls, so the total cost is 24 * $15 = $370.

Now, let's add up all the costs to find out how much the market will have received after all the balls are sold:

Total cost of balls - Total cost of yellow balls - Total cost of green balls
$110 - $150 - $370

This is the final answer:

$110 + $150 + $370 = $560

The market will have received $560 after all the balls are sold."""

import string


def last_line_parser(inp):
    last_line = list(
        filter(
            lambda x: len(x.strip().strip(string.punctuation).strip()) > 0,
            inp.split("\n"),
        )
    )
    if last_line:
        last_line = last_line[-1].strip().lower()
        return last_line
    return ""


last_line = last_line_parser(ans)
digits = re.findall(r"[+-]?\d+", last_line)
if digits:
    print(digits[-1])


# In[ ]:


import re

pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?"
text = "Values are -3, +4.5, .25, 2/3, -1.5/4"

print([eval(x) for x in re.findall(pattern, text)])


# In[ ]:


def linear_decay_with_warmup(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
):
    assert total_steps > warmup_steps + decay_steps

    def schedule(step):
        # Convert step to tensor if it's a scalar
        if isinstance(step, (int, float)):
            step_tensor = torch.tensor(step, dtype=torch.float32)
        else:
            step_tensor = step.float()

        # Linear warmup: 0 → base_lr
        warmup_lr = base_lr * step_tensor / warmup_steps
        # Linear decay: base_lr → 0
        decay_progress = (step_tensor - (total_steps - decay_steps)) / decay_steps
        decay_lr = base_lr * (1.0 - decay_progress)

        lr = torch.where(
            step_tensor < warmup_steps,
            warmup_lr,
            torch.where(
                step_tensor >= (total_steps - decay_steps),
                torch.clamp(decay_lr, min=0.0),
                torch.tensor(
                    base_lr, device=step_tensor.device, dtype=step_tensor.dtype
                ),
            ),
        )
        return lr

    return schedule


# In[ ]:


import matplotlib.pyplot as plt

# Scheduler config
base_lr = 1e-5
total_steps = 500
warmup_steps = 50
decay_steps = 50

scheduler = linear_decay_with_warmup(base_lr, total_steps, warmup_steps, decay_steps)

# Compute LR values
steps = list(range(total_steps))
lrs = [float(scheduler(s).item()) for s in steps]

# Plot
plt.figure()
plt.plot(steps, lrs)
plt.grid(True)
plt.xlabel("Training step")
plt.ylabel("Learning rate")
plt.title("Linear Warmup + Linear Decay LR Schedule")
plt.show()


# In[ ]:
