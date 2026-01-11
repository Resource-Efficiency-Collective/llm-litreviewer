def run(args):
    print("Running in structured mode...")
    # Add your logic here


from articlefilter.filter_class import LLMProcessor_Pure


def run(args):

    print("Running in structured mode...")
    print(f"Config: {args.config}")
    # Add your logic here

    # config = "./config/test_binary_prompt.yaml"
    llm_processor = LLMProcessor_Pure.from_config(
        args.config, logits=True, embedding=False
    )
    # llm_processor.run_grammar()
    llm_processor.runStructured()
