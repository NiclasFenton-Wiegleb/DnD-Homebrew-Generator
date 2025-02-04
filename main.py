import sys
import os
sys.path.append('./pipelines')

from dotenv import load_dotenv, find_dotenv

from module_1 import Module1


if __name__ == '__main__':
    
    load_dotenv(find_dotenv())
    AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    # Module 1 execution
    module_1 = Module1()
    client, assistant = module_1.create(endpoint= AZURE_OPENAI_ENDPOINT, api_key= AZURE_OPENAI_API_KEY)