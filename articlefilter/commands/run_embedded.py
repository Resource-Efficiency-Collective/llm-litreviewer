from articlefilter.filter_class import LLMProcessor_Pure


def run(args):
    print("Running in embedded mode...")
    # Inspired by https://simonwillison.net/2023/Oct/23/embeddings/
    print(f"Config: {args.config}")
    # Add your logic here

    # config = "./config/test_binary_prompt.yaml"
    llm_processor = LLMProcessor_Pure.from_config(
        args.config, logits=False, embedding=True
    )
    llm_processor.runEmbedding(batch_size=10)
