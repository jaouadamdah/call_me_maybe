from .cli import get_args
import sys
import time

if __name__ == "__main__":
    try:
        args = get_args()

        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
        from .app import App

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
                print(f"\n{'-' * 10}[{args.name}]{'-' * 10}\n")
                pipeline = App(
                    model,
                    args.functions_definition,
                    args.max_token if args.max_token > 0 else None,
                )
                t0 = time.perf_counter()
                pipeline.run(args.input, args.output)
                print(time.perf_counter() - t0)
            except (FileNotFoundError, PermissionError, ValueError) as e:
                print(e, file=sys.stderr)
            except Exception as error:
                print(
                    f"An error occurred while running the program:\n{error}",
                    file=sys.stderr,
                )
                pass
    except KeyboardInterrupt:
        exit()
