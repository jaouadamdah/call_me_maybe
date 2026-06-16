import argparse


def get_args() -> argparse.Namespace:
    """Parses and retrieves arguments.

    Returns:
        argparse.Namespace: An object containing the parsed args as attributes.
    """

    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="Run the LLM function calling.",
        usage=(
            "uv run python3 -m src "
            "[--functions_definition <function_definition_file>] "
            "[--input <input_file>] "
            "[--output <output_file>]"
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        metavar="<file>",
        help="Path to the JSON file containing the function schemas.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        metavar="<file>",
        help="Path to the JSON input file containing prompts.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        metavar="<file>",
        help="Path where the generated JSON function calls will be saved.",
    )
    parser.add_argument(
        "--name",
        default="Qwen/Qwen3-0.6B",
        metavar="<model_name>",
        help="The model name to be loaded by the LLM SDK.",
    )
    parser.add_argument(
        "--max_token",
        default=220,
        type=int,
        metavar="<int>",
        help=(
            "The max number of tokens to generate per prompt."
            "(set to 0 or negative to disable the limit)."
        ),
    )
    return parser.parse_args()
