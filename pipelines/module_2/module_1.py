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
        with open('/home/niclaswiegleb/projects/DnD-Homebrew-Generator/pipelines/module_2/output_template.json') as f:
            output_template = json.load(f)

        system_prompt = f'''
            ### Context ###
            You are an AI assisstant that helps write Dungeons and Dragons campaigns.
            You create location descriptions, writes background information and
            imagines a history of the land and its people. This provides context and allows players to
            visualise the locations they visit.

            ### Input ###
            The input you receive will contain the each location relevant to the narrative and any
            important environmental details that need to be included in the relevant description.

            ### Output ###
            Ensure that the descriptions are descriptive and written in an emotive tone that draws in the
            player. The descriptions should be kept short. All clues need to be mentioned in the description.

            locations:  list of locations where the narrative takes place in the order the locations are visited.
            description: description associated with the respective location
            Background: A longer text providing the history and backdrop to the start of the narritive.

            ### Output Template ###

            json:
            {output_template}
            '''
        
        self.sys_prompt = system_prompt