# Call Me Maybe

> A function-calling system that converts natural language requests into structured JSON function calls — with **guaranteed valid output** via constrained decoding.

---

## Overview

**Call Me Maybe** is built on top of a small language model (LLM). Unlike traditional prompting approaches that rely on the model to naturally produce correct JSON — which often results in invalid output, especially with small LLMs — this project uses **constrained decoding** to restrict token generation at every step. The final output is guaranteed to perfectly match the expected JSON structure and the function definitions provided by the user.

---

## Example

**Input:**
```
What is the sum of 40 and 2?
```

**Output:**
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

> **Note:** The system does not execute the function. It only determines which function should be called, which arguments to provide, and the correct argument types.

---

## Features

- Function selection using an LLM
- Schema-aware constrained decoding
- Guaranteed valid JSON generation
- Validation using Pydantic
- Robust error handling
- Command-line interface with configurable input/output and tools schema paths, model name, and max tokens to be generated 

---

## Project Structure

```
.
├── data
│   └── input
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk
│   ├── llm_sdk
│   │   └── __init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── makefile
├── pyproject.toml
├── README.md
├── src
│   ├── app.py
│   ├── cli.py
│   ├── generator.py
│   ├── __init__.py
│   ├── loaders.py
│   ├── __main__.py
│   ├── models.py
│   ├── NextTokenSelector.py
│   └── state_generator.py
└── uv.lock
```

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd call_me_maybe
```

Install dependencies:

```bash
uv sync
```

Or using Make:

```bash
make install
```

---

## Running the Program

**Default execution:**

```bash
uv run python -m src
```

Or:

```bash
make run
```

**Custom input files:**

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json 
```

---

## Development Commands

| Command | Description |
|---|---|
| `make debug` | Run in debug mode |
| `make lint` | Static analysis |
| `make lint-strict` | Strict static analysis |
| `make clean` | Remove temporary files |

---

---
## Command-Line Arguments
| Command | Description |
|---|---|
|-h or --help | Show the help message and exit.|
|--functions_definition | Path to the JSON file containing the function schemas.|
|--input | Path to the JSON input file containing prompts.|
|--output | Path where the generated JSON function calls will be saved.|
|--name | The model name to be loaded by the LLM SDK.|
|--max_token | The max number of tokens to generate per prompt (set to 0 or negative to disable the limit).|
---

## How It Works

### The Core Challenge

The goal is to guarantee that every generated response is valid JSON and follows the exact schema defined in the function definitions. This is achieved through **constrained decoding** — a combination of character-level state machines and logit masking.

### Generation States

A character-level state machine is built from the expected output schema. Each state tracks allowed characters and transitions to the next state.

```python
states = {
  0: {"{": 1},
  1: {"n": 2},
  2: {"a": 3},
  3: {"m": 4},
  4: {"e": 5},
  # ...
}
```

This gives the LLM the freedom to choose any valid token — even multi-character ones — maintaining high generation quality and minimizing total token count.

### Vocabulary Structure

The vocabulary is stored as a dictionary for fast lookup of allowed tokens per state.

```python
vocab = {
  'any': {
    # Token IDs valid for generating string parameter values
  },
  '{': {
    '{': 4,
    '{"': 85,
    '{"name': 1402,
    # All tokens starting with '{'
  },
  # ...
}
```

### Generation Process

1. Encode the prompt into tokens
2. Send tokens to the LLM to generate logits
3. Determine the next token based on the current state:
   - **Cache hit:** Retrieve masked tokens directly
   - **Cache miss:** Walk the vocabulary from the current state, validating each token character by character against the state machine
4. Apply **logit masking** — set disallowed token probabilities to `-inf`
5. Select the highest-probability valid token and transition to the next state
6. Repeat until reaching an end state or the maximum token limit

**Example of multi-character token selection:**

```python
# Instead of forcing single characters:
# '{' -> '"' -> 'p' -> 'a' ...

# The LLM can select multi-character tokens:
# '{"' -> 'parameters' -> '":' -> ' {'
```

### Caching Strategy

- **Normal states** cache allowed token IDs
- **The `any` state** (for dynamic string values) caches *invalid* token IDs instead

This asymmetry significantly reduces masking overhead during generation.

---

## Design Decisions

### Pydantic Validation

All internal structures use Pydantic models, providing:
- Automatic validation
- Clear error messages
- Strong typing
- Easier maintenance and readability

### Error Handling

Errors are handled gracefully with clear, actionable messages. Covered cases include:
- Missing files
- Invalid JSON input
- Invalid function definitions
- Output generation failures

---

## Performance

### Accuracy

Because the LLM retains freedom to choose optimal, multi-character tokens (provided they align with valid state transitions), generation remains natural and highly accurate — without sacrificing schema compliance.

### Reliability

The constrained decoder guarantees:
- 100% valid JSON
- No malformed output or trailing commas
- No missing required fields
- Correct parameter names and types

### Speed

- **NumPy** is used for high-speed during logit masking
- The **caching system** plus dictionary-based vocabulary lookup reduces time spent computing allowed tokens
- **compress base prompt** reduce compute time during the logit generation phase.

---

## Challenges

**Pointer management:** Giving the LLM freedom to choose multi-character tokens while strictly enforcing the state machine required precise pointer tracking through each character of every candidate token.

**Stopping during generating arguments value:** For example when ending a string argument value, the LLM has full freedom to choose any token that contains a closing " — including multi-character tokens like ",, "}, or "}}. The only condition is that every character after the " in that token must remain valid according to the current state machine. Preserving generation quality without ever breaking the schema.

---

## Testing

### Unit Tests

Individual components are tested independently:
- JSON loading
- Schema validation
- State generator
- Logit masking logic
- Output generation

### Edge Cases

- Empty prompts and empty strings
- Exceptionally large numbers
- Invalid or missing JSON files
- Incorrect parameter types in definitions
- Ambiguous natural language prompts

---

## Example Usage

**Function definition:**

```json
{
  "name": "fn_reverse_string",
  "description": "Reverse a string",
  "parameters": {
    "s": {
      "type": "string"
    }
  }
}
```

**Prompt:**
```
Reverse the string 'hello'
```

**Generated output:**

```json
{
  "prompt": "Reverse the string 'hello'",
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello"
  }
}
```

---

## Resources

**Generale**
- [How LLM Works](https://youtu.be/5sLYAQS9sWQ?si=Qe1YSjIMjE2En0q4)
- [LLMs Fine-Tuning](https://www.youtube.com/live/IYVV0boR_DI?si=oUn2tHRv5KsJv6pb)

**Function Calling**
- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [Hugging Face Function Calling Documentation](https://huggingface.co/docs/hugs/en/guides/function-calling)
- [Pydantic Documentation](https://docs.pydantic.dev)

**Constrained Decoding**
- [Implementing Constrained Decoding](https://medium.com/@albersj66/part-6-implementing-constrained-decoding-for-phi-3-vision-2c72a1be6a17)
- [Grammar-Guided Generation for Structured LLM Output](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)

**Python**
- [Python Documentation](https://docs.python.org/3/)

---

## AI Usage

AI tools were used for understanding constrained decoding concepts, researching JSON schema validation techniques, and help in docstring and readme file.