*This project has been created as part of the 42 curriculum by jamdah.*

# call_me_maybe

## Description
**call me maybe** is a Python project, it's main goal is to transform small Large Language Models (LLMs) into reliable, structured engines. While LLMs default to generating natural language and frequently struggle to produce valid JSON especially smaller models, this project implements a constrained decoding pipeline that manages the entire generation process. By guiding token selection step-by-step, it forces the model to select the correct tool from a provided schema and generate 100% syntactically valid JSON ready for system execution.


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
