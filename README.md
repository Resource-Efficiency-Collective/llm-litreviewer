# llm-litreviewer

LLM Reviewer supports the following filtering types:
- Pure Prompt
- Binary Probability
- Structured Probability
### Interfaces
LLM Reviewer can be used directly via the python library or through the CLI

## Inputs
LLM Reviewer requires a csv with a column containing abstracts.
- A CSV with a column containing abstracts
- System and/or user prompts.  This must contain {abstract}, where the abstract will be inserted.\*
- Configuration file (or individually set model parameters)
- Grammar File (optional)


### Input CSV
While the input CSV must contain a column with abstracts, it can also contain other columns (which can be saved into the final result).  Columns to keep can be specified in the output YAML config section under "keep\_columns" 

### System and user prompts
The prompt should contain relevance criteria, and additionally expected response format.  For the pure prompt approach this is:
```
Label: <Relevance>
Explanation: <Response>
etc...
```
For the binary probability approach a single token (usually 0 or 1) is returned, this must be explicitly specified in the prompt instructions.

If using a grammar file, output format should not be set in the response itself (see grammar).

### Grammar
A grammar file is used to constrain the LLM output to a certain set of tokens.  It is specified in a specified in ".gbnf" format.

Here is a sample .gbnf file:
```
root ::= AbstractInfo
AbstractInfo ::= "{" ws "\"relevance\":" ws boolean "," ws "\"explanation\":" ws string "}"
AbstractInfolist ::= "[]" | "[" ws AbstractInfo ("," ws AbstractInfo)* ws "]"

string ::= "\"" ([^"]*) "\""
boolean ::= "0" | "1"
ws ::= [ \t\n]*
number ::= [0-9]+ "."? [0-9]*
stringlist ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
numberlist ::= "[" ws "]" | "[" ws number ("," ws number)* ws "]"
```

This file specifies the response must be in JSON format, and that it consists of 2 entries. Relevance, which is a boolean value (either 0 or 1) and an explanation which is a string.

Note that if a .gbnf file is used without specifying the response type in the prompt, the resulting tokens will follow the form, but have extremely low probabilities.  To prevent this, grammar is specified in a pair of files (grammar.gbnf and grammar\_explanation.txt).  These are placed a directory which is specified when configuring the model.

The accompanying grammar explanation might appear as:
```
Generate a structured JSON object following this schema: { "relevance": boolean, "explanation": string }. Ensure boolean values are either 0 or 1, the decision should be explained concisely.
```

### Configuration file
The configuration for a model can be loaded from a YAML file (either in python or through the CLI interface). A sample configuration file looks like this: 
```
model:
  name: "llama3.2"
  version: null  # or you can just omit this line if you prefer
  # provider: "ollama"
  provider: "llama_cpp"
  llama_cpp_settings:
    max_context_size: 2048
    n_ctx: 2048
    verbose: False
    n_threads: 4
    simple: True


evalulation:
  max_tokens: 1
  top_p: 1
  temperature: 0
  presence_penalty: 0
  frequency_penalty: 0
  repeat_penalty: 1


abstracts:
  input_file: "./input/combined_filtered_type.csv"

prompts:
  # system_prompt_path: "./system_messages/prompt.txt"
  user_prompt_path: "./user_messages/probs_input_3.txt" 

output:
  output_dir: "./output"
  run_name: "test"
  keep_columns:
    - "Authors"
    - "Article Title"
    - "UT (Unique WOS ID)"
  relevance_label_col_name: 'Label'
  # other_cols:
  #   - "Explanation"

# Optional:
log_stats: false  # change to true if you want logging

```
The configuration file is split into sections for model, evaluation, abstracts, prompts, and output.

### Python Library

First, load the LLMProcessor Class, this loads the LLM model and specifies settings.

Models installed via ollama can be accessed by specifying the model and version.


```python
# Example:

model_name = "llama3.2"
model_version = None
model_provider = "ollama"
input_file = "./input/combined_filtered_type.csv"
output_dir = "./output"
run_name = "test"
user_prompt_path = "./user_messages/user_message_template.txt"
system_prompt_path = "./system_messages/prompt.txt"
keep_columns = ["Authors", "Article Title", "UT (Unique WOS ID)"]

llm_processor = LLMProcessor_Pure()
llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
)

llm_processor.load_abstracts(input_file=input_file)
llm_processor.load_prompt(
    user_prompt_path=user_prompt_path, system_prompt_path=system_prompt_path
)
llm_processor.prepare_output_files(
    output_dir=output_dir, run_name=run_name, keep_columns=keep_columns
)
llm_processor.run()

```
Alternatively, if your parameters are in a yaml file, you can simply use:
```python
config = "./config/test_pure_prompt.yaml"
llm_processor = LLMProcessor_Pure.from_config(config)
llm_processor.run()
```
### CLI

If you've installed via poetry, prefix the command with the poetry run command. 
```
articlefilter runPure --config path_to_yaml
articlefilter runBinary --config path_to_yaml
articlefilter runStructured --config path_to_yaml
articlefilter runEmbedded --config path_to_yaml
```


## Embedding Methods
LLMs assign a highdimensional vector to each token of the input.  If we take the cosine distance between vectors of two tokens we get their similarity.  For example, Dog is likely closer to cat than it is to airplane.

Embedding models return the embedding vector for the entire input (instead of one for each token).  If we calculate the embedding vector for each abstract, we can compare and cluster them.  We can even vectorize a query say "Gray Whale Behavior" and find the abstracts that are most relevant based on the cosine distance.

For more information, Simon Willson has a great write up on his blog: https://simonwillison.net/2023/Oct/23/embeddings/

We implement this method for abstracts using the llama\_cpp library.
```
# Load Model
model_name = "nomic-embed-text"
model_version = None
model_provider = "llama_cpp"
input_file = "../input/combined_filtered_type.csv"
llm_processor = af.LLMProcessor_Pure()
llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    embedding=True,
)

llm_processor.runEmbedding(batch_size=10)

# To send in text directly

sentences = [
    "The cat sat.",
    "The cat ran.",
    "A cat ran.",
    "The bank is up hill.",
    "FLY!",
    "Me gusta la ciudad.",
]
res = llm_processor.model_provider.llm.create_embedding(sentences)

```
