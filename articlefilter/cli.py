import argparse
from articlefilter.commands import run_pure, run_binary, run_structured, run_embedded


def main():
    parser = argparse.ArgumentParser(
        description="Run initial screening of abstracts using different LLM evaluation modes."
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    # runPure command
    parser_pure = subparsers.add_parser("runPure", help="Run the pure LLM filter.")
    parser_pure.add_argument("--config", type=str, help="Path to config YAML.")
    parser_pure.set_defaults(func=run_pure.run)

    # runBinary command
    parser_binary = subparsers.add_parser(
        "runBinary", help="Run binary evaluation mode."
    )
    parser_binary.add_argument("--config", type=str, help="Path to config YAML.")
    parser_binary.set_defaults(func=run_binary.run)

    # runStructured command
    parser_structured = subparsers.add_parser(
        "runStructured", help="Run structured reasoning mode."
    )
    parser_structured.add_argument("--config", type=str, help="Path to config YAML.")
    parser_structured.set_defaults(func=run_structured.run)

    # runEmbedding command
    parser_structured = subparsers.add_parser(
        "runEmbedding", help="Run Embedded Filtering."
    )
    parser_structured.add_argument("--config", type=str, help="Path to config YAML.")
    parser_structured.set_defaults(func=run_embedded.run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
# def run_pure(args):
#     print("Running in pure mode...")
#     # Add your logic here
#
# def run_binary(args):
#     print("Running in binary mode...")
#     # Add your logic here
#
# def run_structured(args):
#     print("Running in structured mode...")
#     # Add your logic here
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Run initial screening of abstracts using different LLM evaluation modes."
#     )
#     subparsers = parser.add_subparsers(title="Commands", dest="command")
#     subparsers.required = True
#
#     # runPure command
#     parser_pure = subparsers.add_parser("runPure", help="Run the pure LLM filter.")
#     parser_pure.add_argument("--config", type=str, help="Path to config YAML.")
#     parser_pure.set_defaults(func=run_pure)
#
#     # runBinary command
#     parser_binary = subparsers.add_parser("runBinary", help="Run binary evaluation mode.")
#     parser_binary.add_argument("--config", type=str, help="Path to config YAML.")
#     parser_binary.set_defaults(func=run_binary)
#
#     # runStructured command
#     parser_structured = subparsers.add_parser("runStructured", help="Run structured reasoning mode.")
#     parser_structured.add_argument("--config", type=str, help="Path to config YAML.")
#     parser_structured.set_defaults(func=run_structured)
#
#     args = parser.parse_args()
#     args.func(args)
#
# if __name__ == "__main__":
#     main()
