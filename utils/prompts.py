PROMPTS = {
    'llama_chat': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    'prompt': """Below is a question I would like you to answer.

Quesiton: {question}

First give a concise step by step breakdown that is numbered and newline delimited. Give only one thought per line. Then, choose a single final answer{options}

IMPORTANT: Please strictly adhere to the following format:

The start of Assistant's step by step breakdown
<give a concise step by step breakdown that is numbered and newline delimited>
Assistant's final answer: <provide a single final answer>"""
}

mistral_mc = """<s>[INST] Please answer the Question below with the following method:
Question: {prompt}
First give a concise step by step breakdown that is numbered and newline delimited.
Give only one thought per line.
On the last line, choose a single Final Answer{extra_for_mc}
{answ_opt}
I will extract the Final Answer from the last line with regex. [/INST] """

mistral_no_a = """<s>[INST] Please answer the Question below with the following method:
Question: {prompt}
First give a concise step by step breakdown that is numbered and newline delimited.
Give only one thought per line.
On the last line, choose a single Final Answer{extra_for_mc}
I will extract the Final Answer from the last line with regex. [/INST] """

old_no_a = """<s>[INST] Please answer the Question below with the following method:
Question: {prompt}
First give a concise step by step breakdown that is numbered and newline delimited.
Give only one thought per line.
On the last line, give a single, concise Final Answer.
I will extract the Final Answer from the last line with regex. [/INST] """