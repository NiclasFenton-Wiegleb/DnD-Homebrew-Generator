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

    def load_prompt(self):
        # System prompts and output templates should be kept separate and each module should
        # have its own version that is initialised with the module.
        with open('/home/niclaswiegleb/projects/DnD-Homebrew-Generator/pipelines/base_module/output_template.json') as f:
            output_template = json.load(f)

        system_prompt = f'''### Context ###
            You are an AI assisstant that extracts the locations, characters, key actions and clues of a narrative.

            ### Input ###
            You receive a narrative which contains different locations, key characters, actions and clues which move the narrative forward.

            ### Output ###

            Only include information vital to the plot. Ignore information that does not advance the narrative.

            locations:  list of locations where the narrative takes place in the order the locations are visited.
            characters: list of characters that appear in the narrative and the location where they first appear.
            actions: list of actions that advance the narrative. This includes the characters taking the action and the location where the action occurs.
            clues: list of clues that are observed by the protagonist that move the narrative forward. This includes the location of the clue and a summary of the significance of the clue.

            ### Output Template ###

            json:
            {output_template}
            '''
        
        self.sys_prompt = system_prompt

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