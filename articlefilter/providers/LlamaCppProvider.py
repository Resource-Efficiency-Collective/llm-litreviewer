import os
import json
from llama_cpp import Llama, LlamaGrammar


class LlamaCppProvider:
    def __init__(
        self,
        model_name,
        model_version=None,
        logits=False,
        embedding=False,
        grammar_file=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_version = model_version if model_version else "latest"
        self.logits = logits
        self.embedding = embedding
        # if grammar_file:
        #     self.grammar = LlamaGrammar.from_file(grammar_file)
        # else:
        #     self.grammar = None

        # Load the model based on whether a GGUF file path is provided or not
        verbose = kwargs.get("verbose", "NOT SET")
        print(f"Verbose: {verbose}")
        print(f"kwargs: {kwargs}")
        if os.path.exists(model_name):  # If model_name is a path to a GGUF file
            self.llm = self.load_model_from_path(model_name, **kwargs)
        else:
            self.llm = self.load_ollama_model(model_name, self.model_version, **kwargs)
            print("MODEL LOADED")

    def load_ollama_model(
        self,
        model,
        version="latest",
        simple=True,
        **kwargs,
    ):
        """Load model from Ollama directory."""

        print(f"Simple: {simple}")
        # Construct the manifest path for the model
        manifest_path = (
            f"~/.ollama/models/manifests/registry.ollama.ai/library/{model}/{version}"
        )
        print(manifest_path)
        manifest_path = os.path.expanduser(manifest_path)

        try:
            with open(manifest_path, "r") as file:
                data = json.load(file)

            # Extract model blob (first layer with "application/vnd.ollama.image.model")
            model_blob = next(
                layer["digest"]
                for layer in data["layers"]
                if layer["mediaType"] == "application/vnd.ollama.image.model"
            )

            model_blob = model_blob.replace(":", "-")  # Replace : with - in model_blob
            blob_path = f"~/.ollama/models/blobs/{model_blob}"  # GGUF file path
            blob_path = os.path.expanduser(blob_path)
            # The default n_ctx should be 2048
            print(kwargs)

            default_settings = {
                "logits_all": self.logits,
                "embedding": self.embedding,
                "verbose": False,
                "n_threads": 1,
                "n_ctx": 2048,
                "max_context_size": 2048,
                # "grammar": self.grammar,
            }
            print(f"Model Settings: {default_settings}")

            # Merge defaults with passed kwargs
            settings = {**default_settings, **kwargs}

            print(settings)
            # logits_all = self.logits_all
            llm = Llama(
                model_path=blob_path,
                **settings,
                # ,chat_format='llama-2'
            )

            print(f"Llama model loaded from Ollama registry: {model}")
            return llm
        except FileNotFoundError:
            raise ValueError(f"Model manifest not found for {model} version {version}")

    def load_model_from_path(self, model_path, simple=True):
        """Load the model directly from a GGUF file."""
        model_path = os.path.expanduser(model_path)  # Expand user in the path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at {model_path}")

        # Initialize Llama with the GGUF model path
        llm = Llama(model_path=model_path)
        # if simple == True:
        #     llm = Llama(
        #         model_path=model_path,
        #         n_ctx=2048,
        #         # ,chat_format='llama-2'
        #     )
        # else:
        #
        #     llm = Llama(
        #         model_path=model_path,
        #         logits_all=True,
        # max_context_size=2048,
        # n_ctx=2048,
        #
        # max_context_size=10000,
        # n_ctx=10000,
        # verbose=False,
        # n_threads=4,
        # )

        print(f"Llama model loaded from GGUF file: {model_path}")
        return llm

    def chat(
        self,
        system_message,
        user_message,
        # max_tokens=150,
        max_tokens=2000,
        logprobs=None,
        top_p=1.0,
        temperature=0.8,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repeat_penalty=1.0,
    ):
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        # Assuming the response object contains a field 'text' with the generated content
        # print(response)
        return response["choices"][0]["message"]["content"].strip(), response

    def get_embedding(
        self,
        input_text,
        **kwargs,
    ):
        # print(f"Default Settings: {default_settings}")
        # Merge defaults with passed kwargs
        response = self.llm.create_embedding(
            input_text,
        )

        return response

    def _load_grammar(self, grammar_file):
        return LlamaGrammar.from_file(grammar_file)

    def _load_grammar_string(self, grammar_string):
        return LlamaGrammar.from_string(grammar_string)

    def calc_grammar(
        self,
        input_text,
        grammar=None,
        grammar_string=None,
        **kwargs,
    ):
        # print(kwargs)
        if grammar:
            grammar = self._load_grammar(grammar)
        elif grammar_string:
            grammar = self._load_grammar_string(grammar_string)

        response = self.llm(
            input_text,
            grammar=grammar,
            **kwargs,
        )

        result_text = response["choices"][0]["text"]
        return result_text, response

    def boolean_probs(
        self,
        input_text,
        **kwargs,
    ):
        # Set up defaults
        default_settings = {
            "max_tokens": 1,
            "logprobs": 3,
            "top_p": 1.0,
            "temperature": 0.8,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
        }
        # print(f"Default Settings: {default_settings}")
        # Merge defaults with passed kwargs
        settings = {**default_settings, **kwargs}
        # print(settings)
        response = self.llm(
            input_text,
            grammar=None,
            **settings,
        )

        # response = self.llm(
        #     input_text,
        #     grammar=None,  # Assuming grammar can be passed here if needed
        #     max_tokens=max_tokens,
        #     logprobs=logprobs,
        #     top_p=top_p,
        #     temperature=temperature,
        #     presence_penalty=presence_penalty,
        #     frequency_penalty=frequency_penalty,
        #     repeat_penalty=repeat_penalty,
        # )
        result = response["choices"][0]["text"]
        return result, response

    # def get_embeddings(
    #     self,
    #     input_text,
    #     **kwargs,
    # ):
    #     # Set up defaults
    #     default_settings = {
    #         "max_tokens": 1,
    #         "logprobs": 3,
    #         "top_p": 1.0,
    #         "temperature": 0.8,
    #         "presence_penalty": 0.0,
    #         "frequency_penalty": 0.0,
    #         "repeat_penalty": 1.0,
    #     }
    #     # print(f"Default Settings: {default_settings}")
    #     # Merge defaults with passed kwargs
    #     settings = {**default_settings, **kwargs}
    #     response = self.llm(
    #         input_text,
    #         grammar=None,
    #         **settings,
    #     )
    #
    #     result = response["choices"][0]["text"]
    #     return result, response
