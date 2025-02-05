import sys
import os
sys.path.append('./pipelines')
sys.path.append('./data_loaders')

from dotenv import load_dotenv, find_dotenv
from azure.identity import DefaultAzureCredential


from module_1 import Module1
from data_loader import DataLoader


if __name__ == '__main__':
    
    # Configure environmental variables
    load_dotenv(find_dotenv())

    # DataLoader variables
    client_id = os.environ.get('AZURE_CLIENT_ID')
    tenant_id = os.environ.get('AZURE_TENANT_ID')
    client_secret = os.environ.get('AZURE_CLIENT_SECRET')

    # Module variables
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    api_key = os.environ.get('AZURE_OPENAI_API_KEY')

    model="gpt-35-turbo" # replace with model deployment name.
    
    account_url = "https://dndblobdata.blob.core.windows.net"
    credential = DefaultAzureCredential()

    data_loader = DataLoader(account_url, credential)

    data_loader.create()

    container_name='grimms_tales'
    filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    filename = 'grimms_tale.xlsx'

    data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)

    # Module 1 execution
    # module_1 = Module1()
    # module_1.load_prompt()
    # module_1.create(endpoint= endpoint, api_key= api_key, model= model)
    # print(module_1.assistant.id)