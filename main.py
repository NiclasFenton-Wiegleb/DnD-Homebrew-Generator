import sys
import os
sys.path.append('./pipelines')
sys.path.append('./data_loaders')

from dotenv import load_dotenv, find_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient


from module_1 import Module1
from data_loader import DataLoader


if __name__ == '__main__':
    
    # Configure environmental variables
    load_dotenv(find_dotenv())

    # DataLoader variables
    # Service Principle
    client_id = os.environ.get('AZURE_CLIENT_ID')
    tenant_id = os.environ.get('AZURE_TENANT_ID')
    client_secret = os.environ.get('AZURE_CLIENT_SECRET')
    # Key vault
    vault_url = os.environ.get('AZURE_VAULT_URL')
    # Blob storage
    account_url = os.environ.get('AZURE_STORAGE_URL')
    # AI Search
    ai_search_endpoint = os.environ.get('AZURE_SEARCH_SERVICE_ENDPOINT')
    index_name = os.environ.get('AZURE_SEARCH_INDEX_NAME')

    secret_name = '__'

    # create a credential
    credentials = ClientSecretCredential(
        client_id = client_id,
        client_secret = client_secret,
        tenant_id = tenant_id
    )

    # # create a secret client
    # secret_client = SecretClient(vault_url = vault_url, credential= credentials)

    # # retrieve the secret vaule from key vault
    # secret = secret_client.get_secret(secret_name)
    # print('The secret value is :' + secret.value)

    # Module variables
    # endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    # api_key = os.environ.get('AZURE_OPENAI_API_KEY')

    # model="gpt-35-turbo" # replace with model deployment name.
    
    
    # credential = DefaultAzureCredential()

    data_loader = DataLoader(account_url, credentials)

    data_loader.create()

    container_name='grimms-tales'
    filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    filename = 'grimms_tale.xlsx'

    data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)

    # Module 1 execution
    # module_1 = Module1()
    # module_1.load_prompt()
    # module_1.create(endpoint= endpoint, api_key= api_key, model= model)
    # print(module_1.assistant.id)