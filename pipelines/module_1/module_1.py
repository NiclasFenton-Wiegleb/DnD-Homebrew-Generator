import json
import requests
import time

from openai import AzureOpenAI

from base_module import BaseModule

class Module1(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)

    def load_prompt(self):
        # System prompts and output templates should be kept separate and each module should
        # have its own version that is initialised with the module.
        # Use the whole file path for the output.json file.
        with open('/home/niclaswiegleb/projects/DnD-Homebrew-Generator/pipelines/module_1/output_template.json') as f:
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