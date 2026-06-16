import json
from typing import Any
from pydantic import ValidationError

from .models import ToolSet, PromptSet, Vocab


def _load_json_file(path: str) -> Any:
    """Reads and parses a JSON file.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        Any: The parsed JSON data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file lacks read permissions.
        ValueError: If the file contains invalid JSON or another unexpected
        error occurs.
    """

    try:
        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Error: File not found: "{path}"')
    except PermissionError:
        raise PermissionError(
            f'Error: Permission denied. Cannot read the file: "{path}"'
        )
    except json.JSONDecodeError as e:
        raise ValueError(f'Error: Invalid JSON in file "{path}": {e}')
    except Exception as e:
        raise ValueError(f"An error occurred while reading '{path}': {e}")


def load_functions_schema(path: str) -> dict[str, Any]:
    """Loads, validates, and builds the tool schema from a JSON file.

    Args:
        path (str): The file path to the functions definition JSON.

    Returns:
        dict[str, Any]: A dictionary representing the validated and formatted\
    tools.

    Raises:
        ValueError: If the loaded JSON fails Pydantic validation.
    """

    raw_data: list[Any] = _load_json_file(path)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]
    try:
        return ToolSet(tools=raw_data).build_tools()
    except ValidationError as error:
        raise ValueError(
            f"Error while loading tools schema:\n{_format_error(error)}",
        )


def load_prompts(path: str) -> list[dict[str, str]]:
    """Loads and validates the prompts from a JSON file.

    Args:
        path (str): The file path to the prompts JSON.

    Returns:
        list[dict[str, str]]: A validated list of dict containing prompts.

    Raises:
        ValueError: If the loaded JSON fails Pydantic validation.
    """

    raw_data: list[Any] = _load_json_file(path)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]
    try:
        PromptSet(prompts=raw_data)
    except ValidationError as error:
        raise ValueError(
            f"Error while loading prompts:\n{_format_error(error)}",
        )
    return raw_data


def load_vocab(path: str) -> dict[str, int]:
    """Loads and validates the vocabulary.

    Args:
        path (str): The file path to the vocabulary JSON.

    Returns:
        dict[str, int]: A validated dictionary.

    Raises:
        ValueError: If the loaded JSON fails Pydantic validation.
    """

    raw_data: dict[str, int] = _load_json_file(path)

    try:
        return Vocab(items=raw_data).items
    except ValidationError as error:
        raise ValueError(f"Error while loading vocab:\n{_format_error(error)}")


def _format_error(
    error: ValidationError,
) -> str:
    """Formats Pydantic ValidationErrors into clear error message.

    Args:
        error (ValidationError): The error object caught from Pydantic.

    Returns:
        str: Formatted and clear error message string.
    """

    msg = []
    for e in error.errors():
        err_loc = ".".join(str(loc) for loc in e["loc"])
        prefix = f"{err_loc}:\n  "
        err_type = e["type"]

        if err_type == "missing":
            msg.append(prefix + "Missing required field")
        elif err_type == "value_error":
            msg.append(prefix + f"{e['msg'].replace('Value error, ', '')}")
        elif err_type == "too_short":
            msg.append(prefix + "At least 1 item must be provided.")
        else:
            msg.append(prefix + e["msg"])
    return "\n".join(msg)
