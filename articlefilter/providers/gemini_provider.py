from google import genai


# class GeminiProvider:
#     def __init__(self, model_name, api_key=None):
#         """
#         model_name: e.g. "gemini-2.0-flash"
#         api_key: your Google API key (optional if already set in environment)
#         """
#         self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
#         self.model_name = model_name
#
#     def chat(self, system_message, user_message, **kwargs):
#         # Gemini doesn’t use role-based messages in the same way as OpenAI,
#         # so we merge the system and user content into the prompt.
#         contents = [
#             {"role": "system", "parts": [{"text": system_message}]},
#             {"role": "user", "parts": [{"text": user_message}]},
#         ]
#
#         response = self.client.models.generate_content(
#             model=self.model_name, contents=contents, **kwargs
#         )
#
#         return response.text.strip(), response
class GeminiProvider:
    def __init__(self, model_name, api_key=None):
        """
        model_name: e.g. "gemini-2.0-flash"
        api_key: Google API key (optional if already set in environment)
        """
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model_name = model_name

    def chat(self, system_message, user_message, **kwargs):
        # Combine system instructions with the user input
        combined_prompt = f"{system_message.strip()}\n\n{user_message.strip()}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": combined_prompt}]}],
            **kwargs,
        )

        return response.text.strip(), response
