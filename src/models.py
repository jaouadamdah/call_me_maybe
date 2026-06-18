from typing import Any
from pydantic import BaseModel, Field, model_validator


class ValueType(BaseModel):
    """Represents the expected type of a parameter or return value."""

    typ: str = Field(alias="type")


class ToolSchema(BaseModel):
    """Defines the structure and validation rules for tool"""

    name: str
    description: str = Field(min_length=10)
    parameters: dict[str, ValueType]
    returns: ValueType

    @staticmethod
    def is_name_valid(name: str) -> bool:
        """Validates if a given string is a valid Python identifier.

        Args:
            name (str): The name of the function or parameter.
            length (int): The length of the name.

        Returns:
            bool: True if valid, False otherwise.
        """

        length: int = len(name)
        if length <= 0 or (not name[0].isalpha() and name[0] != "_"):
            return False
        for i in range(1, length):
            if not name[i].isalnum() and name[i] != "_":
                return False
        return True

    @model_validator(mode="after")
    def validate_data(self) -> "ToolSchema":
        """Verify function and parameter names.

        Returns:
            ToolSchema: The validated instance.

        Raises:
            ValueError: If the function/parameter names are invalid or empty.
        """

        if not self.name.strip():
            raise ValueError("Function name cannot be empty")
        if not self.is_name_valid(self.name):
            raise ValueError(f"invalid function name: '{self.name}'")

        for name in self.parameters:
            if not name.strip():
                raise ValueError("Parameter name cannot be empty")
            if not self.is_name_valid(name):
                raise ValueError(f"invalid parameter name: '{name}'")
        return self


class ToolSet(BaseModel):
    """Container for a list of tools loaded from the JSON schema."""

    tools: list[ToolSchema] = Field(min_length=1)

    def build_tools(self) -> dict[str, Any]:
        """Transforms the parsed tools into a simplified dictionary format.

        Returns:
            dict[str, Any]: A mapping of tool names to their descriptions\
                and simplified parameter type mappings.
        """

        return {
            tool.name: {
                "parameters": {arg: typ.typ for arg, typ in
                               tool.parameters.items()},
                "description": tool.description,
            }
            for tool in self.tools
        }


class Prompt(BaseModel):
    """Represents a input prompt."""

    prompt: str

    @model_validator(mode='after')
    def check_prompt(self) -> "Prompt":
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        return self


class PromptSet(BaseModel):
    """Container for a list of prompts."""

    prompts: list[Prompt] = Field(min_length=1)


class Vocab(BaseModel):
    """Represents the vocabulary mapping strings to IDs."""

    items: dict[str, int]

    @model_validator(mode="after")
    def check_id(self) -> "Vocab":
        """Cleans and validates vocabulary items.

        Ensures all token IDs are positive integers and replaces the
        tokenizer's special 'Ġ' space character with a standard space.

        Returns:
            Vocab: The cleaned vocabulary instance.

        Raises:
            ValueError: If any token ID is negative.
        """

        whitespace = {
            '\u0120': ' ',
            '\u010a': '\\n',
            '\u0109': '\\t',
            '\u010d': '\\r',
            '\u010b': '\\v',
            '\u010c': '\\f'
        }

        new_vocab: dict[str, int] = {}
        for token, token_id in self.items.items():
            if token_id < 0:
                raise ValueError(
                    f"Token id must be positive number: {token}:{token_id}"
                )
            new_token = ""
            for char in token:
                new_token += whitespace.get(char, char)

            new_vocab[new_token] = token_id

        self.items = new_vocab
        return self
