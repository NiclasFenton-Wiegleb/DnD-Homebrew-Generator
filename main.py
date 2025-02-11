import sys
import os
sys.path.append('./pipelines')
sys.path.append('./data_loaders')

from dotenv import load_dotenv, find_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import AzureOpenAIVectorizerParameters
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration
)

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
    ai_search_key = os.environ.get('AZURE_SEARCH_SERVICE_KEY')

    secret_name = '__'

    # Tutorial env variables
    # Configuration
    AZURE_AI_STUDIO_COHERE_API_KEY = os.getenv("AZURE_AI_STUDIO_COHERE_API_KEY")
    AZURE_AI_STUDIO_COHERE_ENDPOINT = os.getenv("AZURE_AI_STUDIO_COHERE_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
    ONELAKE_CONNECTION_STRING = os.getenv("ONELAKE_CONNECTION_STRING")
    ONELAKE_CONTAINER_NAME = os.getenv("ONELAKE_CONTAINER_NAME")
    SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")

    # create a credentials
    credentials = ClientSecretCredential(
        client_id = client_id,
        client_secret = client_secret,
        tenant_id = tenant_id
    )

    # User-specified parameter
    USE_AAD_FOR_SEARCH = True  
    # DataLoader
    data_loader = DataLoader(account_url, credentials)
    azure_search_credential = data_loader.authenticate_azure_search(use_aad_for_search=USE_AAD_FOR_SEARCH)
    fields = data_loader.create_fields(1500)
    az_openai_par = AzureOpenAIVectorizerParameters(
                        resource_url=AZURE_OPENAI_ENDPOINT,
                        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
                        api_key=AZURE_OPENAI_API_KEY,
                        model_name=AZURE_OPENAI_MODEL_NAME,
                    )
    index_name = "dnd-generator-openai-index"
    vectorizer_name = "myOpenAI" if "openai" in index_name else None
    print(vectorizer_name)
    vector_search_config = data_loader.create_vector_search_configuration(
        vectorizer_name=vectorizer_name,
        az_openai_parameter=az_openai_par
    )
    print(vector_search_config)
    # # Vector Search
    # vector_search = VectorSearch(
    #     algorithms=[
    #         HnswAlgorithmConfiguration(
    #             name= 'my-vector-config',
    #             parameters={
    #                 'm':4,
    #                 'ef_construction': 400,
    #                 'ef_search': 500,
    #                 'metric': 'cosine'
    #             }
    #         )
    #     ]
    # )
    # # Create a search index client
    # search_client = SearchIndexClient(endpoint=ai_search_endpoint, credential=AzureKeyCredential(ai_search_key))

    # #Create the index
    # #TODO - add embeddings (AzureOpenAIEmbeddings())
    # index_name = 'grimms-tales'
    # fields = [
    #     SimpleField(name='documentID', type=SearchFieldDataType.String, filterable=True, sortable=True, key=True),
    #     SearchableField(name='content', type=SearchFieldDataType.String),
    #     SearchField(name='embedding', type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_profile_name=vector_search, vector_search_dimensions= 1000)
    # ]

    # index = SearchIndex(
    #     name=index_name,
    #     fields=fields,
    #     vector_search=vector_search
    # )

    # result = search_client.create_index(index)

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

    # DataLoader
    # data_loader = DataLoader(account_url, credentials)

    # data_loader.create()

    # container_name='grimms-tales'
    # filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    # filename = 'grimms_tale.xlsx'

    # data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)

    # Module 1 execution
    # module_1 = Module1()
    # module_1.load_prompt()
    # module_1.create(endpoint= endpoint, api_key= api_key, model= model)
    # print(module_1.assistant.id)