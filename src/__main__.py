from .cli import get_args
import sys
import time

if __name__ == "__main__":
    try:
        args = get_args()
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
        from . import app

        print("loading model...")
        try:
            model = Small_LLM_Model(args.name)
        except Exception as e:
            print(
                f"Error: Could not load model '{args.name}':\n{e}",
                file=sys.stderr,
            )
        else:
            try:
                start = time.perf_counter()
                print(f"\n{'-' * 10}[{args.name}]{'-' * 10}\n")
                app.run(
                    model=model,
                    tools_path=args.functions_definition,
                    input_path=args.input,
                    output_path=args.output,
                    max_token=args.max_token if args.max_token > 0 else None,
                )
                print(f'[{(time.perf_counter() - start)/60:.2f} min]')
            except (FileNotFoundError, PermissionError, ValueError) as e:
                print(e, file=sys.stderr)
            except Exception as error:
                print(
                    f"An error occurred while running the program:\n{error}",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        exit()
