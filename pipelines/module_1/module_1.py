import os
import json
import requests
import time
from openai import AzureOpenAI
from dotenv import load_dotenv, dotenv_values

load_dotenv()

class Module1():

    def __init__(self):
       self.client = None
       self.assistant = None
        pass

    def create(self):

        client = AzureOpenAI(
        azure_endpoint = os.getenv(AZURE_OPENAI_ENDPOINT),
        api_key= os.getenv(AZURE_OPENAI_API_KEY),
        api_version="2024-05-01-preview"
        )

        assistant = client.beta.assistants.create(
        model="gpt-35-turbo", # replace with model deployment name.
        instructions='''### Context ###
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
        {
        'locations': [ location1, location2],
        'characters': [{'character_name': character1 , 'location': location1}, {'character_name': character2 , 'location': location2}]
        'actions': [{'action': action1, 'character': character1, 'location': location1}, {'action': action2, 'character': character2, 'location': location2}],
        'clues': [{'clue': clue1, 'location': location1, 'significants': significant_clue1}, {'clue': clue1, 'location': location1, 'significants': significant_clue1}]
        }
        ''',
        tools=[],
        tool_resources={},
        temperature=1,
        top_p=1
        )
        self.client = client
        self.assistant = assistant

        return self.client, self.assistant
