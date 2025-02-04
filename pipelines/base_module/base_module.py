import json
import requests
import time

from openai import AzureOpenAI

from system_prompt import system_prompt


class BaseModule():

    def __init__(self, sys_prompt):
        self.client = None
        self.assistant = None
        self.sys_prompt = sys_prompt

    def create(self, endpoint, api_key, model):

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


with open('output_template.json') as f:
    output_template = json.load(f)

sys_prompt = system_prompt(output_template)
module = BaseModule(sys_prompt)
print(module.sys_prompt)