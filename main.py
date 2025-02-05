import sys
import os
sys.path.append('./pipelines')

from dotenv import load_dotenv, find_dotenv

from module_1 import Module1


if __name__ == '__main__':
    
    load_dotenv(find_dotenv())
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    api_key = os.environ.get('AZURE_OPENAI_API_KEY')
    model="gpt-35-turbo" # replace with model deployment name.

    # Module 1 execution
    module_1 = Module1()
    module_1.load_prompt()
    module_1.create(endpoint= endpoint, api_key= api_key, model= model)
    print(module_1.assistant.id)