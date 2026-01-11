# providers/ollama_provider.py
from ollama import chat


class OllamaProvider:
    def __init__(self, model_name, model_version=None):
        # self.model_name = model_name
        if model_version is not None:
            self.model_name = model_name + f":{model_version}"
        else:
            self.model_name = model_name

    def chat(self, system_message, user_message, **kwargs):
        # print(f"SYSTEM MESSAGE: {system_message}")
        # print(f"USER MESSAGE: {user_message})")
        response = chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            options=kwargs,
        )
        return response["message"]["content"].strip(), response
