import os
import json

from typing import Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .generator import Generator
from .state_generator import StateGenerator
from .NextTokenSelector import NextTokenSelector
from .loaders import load_functions_schema, load_prompts, load_vocab


class App(BaseModel):
    """Main class for function calling process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    tools_path: str
    max_token: int | None = None

    tools: dict[str, Any] = Field(default_factory=dict, init=False)
    vocab: dict[str, int] = Field(default_factory=dict, init=False)

    states: StateGenerator | None = Field(default=None, init=False)
    base_prompt: str = Field(default="", init=False)
    tool_caller: Generator | None = Field(default=None, init=False)

    @model_validator(mode="after")
    def setup_app(self) -> "App":
        """Loads data and initializes all internal components."""

        self.tools = load_functions_schema(self.tools_path)
        self.vocab = load_vocab(self.model.get_path_to_vocab_file())

        self.states = StateGenerator(tools=self.tools)

        self.base_prompt = self._build_base_prompt()

        self.tool_caller = Generator(
            model=self.model,
            states=self.states,
            get_next_token=NextTokenSelector(
                raw_vocab=self.vocab,
                states=self.states,
            ),
            raw_vocab=self.vocab,
            max_token=self.max_token,
        )

        return self

    def _build_base_prompt(self) -> str:
        """Builds the base system prompt containing function definitions.

        Returns:
            str: The formatted system prompt.
        """

        def get_params(params: dict[str, str]) -> str:
            return ",".join(f"{arg}:{typ}" for arg, typ in params.items())

        tools: str = "\n".join(
            f"{nm}({get_params(data['parameters'])}) - {data['description']}"
            for nm, data in self.tools.items()
        )

        return (
            "Tools:\n"
            f"{tools}"
            "\nReturn only JSON:\n"
            '{"name":"function-name","parameters":{"args-name":<args-value>}}'
            "\nUser:\n"
        )

    def run(self, input_path: str, output_path: str) -> None:
        """Processes prompts from the input file and writes results to the\
          output file.

        Args:
            input_path (str): File path to the input JSON containing prompts.
            output_path (str): File path where the generated JSON responses
              will be saved.
        """

        prompts: list[dict[str, str]] = load_prompts(input_path)
        results: list[dict[str, str | dict[str, Any]]] = []
        length: int = len(prompts)

        for i, data in enumerate(prompts, start=1):
            content = data["prompt"]
            print(f"[{i}/{length}] {content}")

            response: str = self.tool_caller.call(
                f"{self.base_prompt} {content}\n",
            )
            try:
                obj: dict[str, str | dict[str, Any]] = json.loads(response)
                results.append(
                    {
                        "prompt": content,
                        "name": obj["name"],
                        "parameters": obj["parameters"],
                    }
                )
                print("[Success] Valid JSON generated.\n")
            except json.JSONDecodeError:
                print("[Failed] Invalid JSON generated.\n")

        if results:
            self.save_responses(output_path, results)
        else:
            print("No responses available to save!")

    @staticmethod
    def save_responses(
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
