# providers/openai_provider.py
from openai import OpenAI
from private_config import api_key


class OpenAIProvider:
    def __init__(self, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def chat(self, system_message, user_message):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip(), response
