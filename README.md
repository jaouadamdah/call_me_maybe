*This project has been created as part of the 42 curriculum by jamdah.*

# call_me_maybe

## Description
**call me maybe** is a Python project, it's main goal is to transform small Large Language Models (LLMs) into reliable, structured engines. While LLMs default to generating natural language and frequently struggle to produce valid JSON especially smaller models, this project implements a constrained decoding pipeline that manages the entire generation process. By guiding token selection step-by-step, it forces the model to select the correct tool from a provided schema and generate 100% syntactically valid JSON ready for system execution.

Unlike traditional prompting approaches that rely on the model to produce correct JSON, this project uses constrained decoding to restrict token generation at every step. This guarantees that the final output matches the expected JSON structure and the function definitions provided by the user.

Example
Input:
```
What is the sum of 40 and 2?
```

Output:
```
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

The system does not execute the function. It only determines:

Which function should be called.

Which arguments should be provided.

The correct argument types.
**Call Me Maybe** is a function-calling system built on top of a small language model (LLM). The goal of the project is to convert natural language requests into structured JSON function calls while guaranteeing that the generated output is always valid and schema-compliant.

## Instructions
# Algorithm Explanation
Our approach to constrained decoding relies on a **states** combined with **logit masking**:

State Generation: The StateGenerator generates states for the provided schema:
```
{"name": <function-name>, "parameters":{<args>}}
```
will generate states for each function schema. it generate signle charater so it convert schema and function schema to characters each state have id, char, and transition id.
example:


Token Filtering: as the llm generate logits , the NextTokenSelector analyzes the current state. It maps vocab and get all allowed tokens that meets the states, when we have token with multiple chars it depands on the sate to move char by char and check it the token meets or not. and each allowed token will have its next state, and cache it so for the next time call we the same state id will get the allowed results directly.
example:


Logit Masking: mask all logits invalid tokens, here we have two apporaches to optimize performance:
   - when we have state 

Selection: The model then selects the most probable token from the remaining valid options, ensuring the output is always structurally and semantically sound.

### Prerequisites
* Python 3.10+
* Virtual environment (recommended)
* Local LLM weights (e.g., Qwen3-0.6B)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd call_me_maybe
   
## Resources



*This project has been created as part of the 42 curriculum by jamdah.*

# Call Me Maybe

## Description
**Call Me Maybe** is a function-calling system built on top of a small language model (LLM). The goal of the project is to convert natural language requests into structured JSON function calls while guaranteeing that the generated output is always valid and schema-compliant.

Unlike traditional prompting approaches that rely on the model to produce correct JSON, that often produce invald JSON specialy small LLMs, this project uses constrained decoding to restrict token generation at every step. This guarantees that the final output matches the expected JSON structure and the function definitions provided by the user.

### Example

Input:
```
What is the sum of 40 and 2?
```
Output:
```
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

The system does not execute the function. It only determines:

- Which function should be called.
- Which arguments should be provided.
- The correct argument types.

### Features

- Function selection using an LLM.
- Schema-aware constrained decoding.
- Guaranteed valid JSON generation.
- Validation using Pydantic.
- Error handling.
- Command-line interface with configurable input/output paths.

### Project Structure
```
.
├── data
│   └── input
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk
│   ├── llm_sdk
│   │   └── __init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── makefile
├── pyproject.toml
├── README.md
├── src
│   ├── app.py
│   ├── cli.py
│   ├── generator.py
│   ├── __init__.py
│   ├── loaders.py
│   ├── __main__.py
│   ├── models.py
│   ├── NextTokenSelector.py
│   └── state_generator.py
└── uv.lock
```

## Instructions
### Installation
Clone the repository:
```
git clone <repository_url>
cd call_me_maybe
```

Install dependencies:
```
uv sync
```
Or:
```
make install
```
### Running the Program

Default execution:
```
uv run python -m src
```
Or:
```
make run
```
Custom input files:
```
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Algorithm Explanation
Our approach to constrained decoding relies on a **states** combined with **logit masking**:

State Generation: The StateGenerator generates states for the provided schema:
```
{"name": <function-name>, "parameters":{<args>}}
```
will generate states for each function schema. it generate signle charater so it convert schema and function schema to characters each state have id, char, and transition id.
example:


Token Filtering: as the llm generate logits , the NextTokenSelector analyzes the current state. It maps vocab and get all allowed tokens that meets the states, when we have token with multiple chars it depands on the sate to move char by char and check it the token meets or not. and each allowed token will have its next state, and cache it so for the next time call we the same state id will get the allowed results directly.
example:


Logit Masking: mask all logits invalid tokens, here we have two apporaches to optimize performance:
   - when we have state 

Selection: The model then selects the most probable token from the remaining valid options, ensuring the output is always structurally and semantically sound.

### Prerequisites
* Python 3.10+
* Virtual environment (recommended)
* Local LLM weights (e.g., Qwen3-0.6B)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd call_me_maybe
   
## Resources
