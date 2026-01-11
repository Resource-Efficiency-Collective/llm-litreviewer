from articlefilter.filter_class import LLMProcessor_Pure


def run(args):
    print("Running in pure mode...")
    # Add your logic here
    # model_name = "llama3.2"
    # model_version = None
    # model_provider = "ollama"
    # input_file = "./input/combined_filtered_type.csv"
    # output_dir = "./output"
    # run_name = "test"
    # user_prompt_path = "./user_messages/user_message_template.txt"
    # system_prompt_path = "./system_messages/prompt.txt"
    # keep_columns = ["Authors", "Article Title", "UT (Unique WOS ID)"]
    #
    # llm_processor = LLMProcessor_Pure()
    # llm_processor.load_model(
    #     model_name=model_name,
    #     model_provider=model_provider,
    #     model_version=model_version,
    # )
    #
    # llm_processor.load_abstracts(input_file=input_file)
    # llm_processor.load_prompt(
    #     user_prompt_path=user_prompt_path, system_prompt_path=system_prompt_path
    # )
    # llm_processor.prepare_output_files(
    #     output_dir=output_dir, run_name=run_name, keep_columns=keep_columns
    # )
    # llm_processor.run()
    # config = "./config/test_pure_prompt.yaml"
    llm_processor = LLMProcessor_Pure.from_config(
        args.config, logits=False, embedding=False
    )
    llm_processor.run()
