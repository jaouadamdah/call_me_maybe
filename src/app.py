import os
import json

from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .generator import Generator
from .state_generator import StateGenerator
from .NextTokenSelector import NextTokenSelector
from .loaders import load_functions_schema, load_prompts, load_vocab


def run(model: Small_LLM_Model,
        tools_path: str,
        input_path: str,
        output_path: str,
        max_token: int | None = None,
        ) -> None:
    """
    Runs the main logic of the application, generating JSON responses based on
    the provided prompts and tools schema.

    Args:
        model (Small_LLM_Model): The LLM to use for generating responses.
        tools_path (str): Path to the JSON file containing the tools schema.
        input_path (str): Path to the JSON file containing the prompts.
        output_path (str): Path to save the generated JSON responses.
        max_token (int | None): Optional maximum number of tokens to generate.
    """

    tools = load_functions_schema(tools_path)
    vocab = load_vocab(model.get_path_to_vocab_file())

    states = StateGenerator(tools=tools)

    base_prompt = _build_base_prompt(tools_schema=tools)

    tool_caller = Generator(
        model=model,
        states=states,
        get_next_token=NextTokenSelector(
            raw_vocab=vocab,
            states=states,
        ),
        raw_vocab=vocab,
        max_token=max_token,
    )

    prompts: list[dict[str, str]] = load_prompts(input_path)
    results: list[dict[str, str | dict[str, Any]]] = []
    length: int = len(prompts)

    for i, data in enumerate(prompts, start=1):
        content = data["prompt"]
        print(f"[{i}/{length}] {content.__repr__()}")

        response: str = tool_caller.call(
            f"{base_prompt}{content}\n",
        )
        try:
            obj: dict[str, Any] = json.loads(response, strict=False)
            _check_output(tools[obj["name"]]["parameters"], obj["parameters"])
            results.append(
                {
                    "prompt": content,
                    "name": obj["name"],
                    "parameters": obj["parameters"],
                }
            )
            print("[Success] Valid JSON generated.\n")
        except json.JSONDecodeError as e:
            print("[Failed] Invalid JSON generated:\n", e)
            print("Generated response:\n", response.__repr__(), "\n")

    if results:
        _save_responses(output_path, results)
    else:
        print("No responses available to save!")


def _check_output(params_schema: dict[str, Any],
                  params_response: dict[str, Any]) -> None:
    """
    Checks the output parameters against the schema and converts them to float
    if necessary.

    Args:
        params_schema (dict): The schema defining the expected parameter types.
        params_response (dict): The generated parameters to validate and
          convert.
    """

    for nm, value in params_response.items():
        if params_schema.get(nm) == "number" and not isinstance(value, float):
            params_response[nm] = float(value)


def _build_base_prompt(tools_schema: dict[str, Any]) -> str:
    """Builds the base system prompt containing function definitions.

    Args:
        tools_schema (dict[str, Any]): The schema defining the available tools
          and their parameters.
    Returns:
        str: The formatted system prompt.
    """

    def get_params(params: dict[str, str]) -> str:
        return ",".join(f"{arg}:{typ}" for arg, typ in params.items())

    tools: str = "\n".join(
        f"{nm}({get_params(data['parameters'])}) - {data['description']}"
        for nm, data in tools_schema.items()
    )

    return (
        "Tools:\n"
        f"{tools}"
        "\nReturn only JSON:\n"
        '{"name":"function-name","parameters":{"args-name":<args-value>}}'
        "\nUser:\n"
    )


def _save_responses(
    path: str,
    res: list[dict[str, str | dict[str, Any]]],
) -> None:
    """Saves the generated JSON responses to the specified path.\
        Ensures the target directory exists before attempting to write.

    Args:
        path (str): File path for the output file.
        results (list[dict[str, str | dict[str, Any]]]): The list of
         parsed JSON objects to save.
    """

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(res, file, indent=2)
            print("Successfully wrote generated responses!")
    except Exception as e:
        print(f"Error occurred while writing file:\n{e}")
