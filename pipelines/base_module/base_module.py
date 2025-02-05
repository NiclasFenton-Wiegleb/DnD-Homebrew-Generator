import json
import requests
import time

from openai import AzureOpenAI

# Base class for subsequent modules (template)
class BaseModule():

    def __init__(self):
        self.client = None
        self.assistant = None
        self.sys_prompt = None



    def create(self, endpoint, api_key, model):
        '''Creates an OpenAI client and assistant based on the system prompt and model specified.'''
        client = AzureOpenAI(
        azure_endpoint = endpoint,
        api_key= api_key,
        api_version="2024-05-01-preview"
        )

        assistant = client.beta.assistants.create(
        model= model,
        instructions= self.sys_prompt,
        tools=[],
        tool_resources={},
        temperature=1,
        top_p=1
        )
        self.client = client
        self.assistant = assistant

if __name__ == '__main__':
    module = BaseModule()
    print(module.sys_prompt)