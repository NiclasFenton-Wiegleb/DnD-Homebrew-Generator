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

def create_search_index(dataloader, USE_AAD_FOR_SEARCH, index_name, vectorizer_name, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_NAME, ai_search_key, SEARCH_SERVICE_ENDPOINT):
    # Create a search index using data_loader
    azure_search_credential = data_loader.authenticate_azure_search(use_aad_for_search=USE_AAD_FOR_SEARCH)
    fields = data_loader.create_fields(1500)
    az_openai_par = AzureOpenAIVectorizerParameters(
                        resource_url=AZURE_OPENAI_ENDPOINT,
                        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
                        api_key=AZURE_OPENAI_API_KEY,
                        model_name=AZURE_OPENAI_MODEL_NAME,
                    )

    vector_search = data_loader.create_vector_search_configuration(
        vectorizer_name=vectorizer_name,
        az_openai_parameter=AzureOpenAIVectorizerParameters(
                        resource_url=AZURE_OPENAI_ENDPOINT,
                        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
                        api_key=AZURE_OPENAI_API_KEY,
                        model_name=AZURE_OPENAI_MODEL_NAME,
                    )
    )
    semantic_search = data_loader.create_semantic_search_configuration()
    ai_search_credentials = AzureKeyCredential(ai_search_key)
    data_loader.create_search_index(
        index_name= index_name,
        fields= fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
        search_service_endpoint= SEARCH_SERVICE_ENDPOINT,
        credential= ai_search_credentials
    )
    print(f"Created index: {index_name}")

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
    index_name = "dnd-generator-openai-index002"
    vectorizer_name = "myOpenAI" if "openai" in index_name else None

    # Module variables
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    api_key = os.environ.get('AZURE_OPENAI_API_KEY')

    model="gpt-35-turbo" # replace with model deployment name.
    # DataLoader
    data_loader = DataLoader(account_url, credentials)
    
    # Create search index
    # create_search_index(
    #     data_loader,
    #     USE_AAD_FOR_SEARCH,
    #     index_name, vectorizer_name,
    #     AZURE_OPENAI_ENDPOINT,
    #     AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    #     AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_NAME,
    #     ai_search_key,
    #     SEARCH_SERVICE_ENDPOINT)

    # Create Skillset
    skillset_name = f"{index_name}-skillset"

    split_skill = data_loader.create_split_skill()
    openai_embedding_skill = data_loader.create_embedding_skill_openai(
        azure_openai_endpoint= AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment= AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
        azure_openai_key= AZURE_OPENAI_API_KEY)
    search_indexer = data_loader.create_index_projections(index_name)
    data_loader.create_indexer_client(SEARCH_SERVICE_ENDPOINT, credential=AzureKeyCredential(ai_search_key))
    skills = [split_skill, openai_embedding_skill]
    data_loader.create_skillset(
        skillset_name=skillset_name,
        skills=skills,
        index_projections=search_indexer
        )
    # Create Blob Storage
    # data_loader.create_blob()

    # container_name='grimms-tales'
    # filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    # filename = 'grimms_tale.xlsx'

    # data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)

    # Module 1 execution
    # module_1 = Module1()
    # module_1.load_prompt()
    # module_1.create(endpoint= endpoint, api_key= api_key, model= model)
    # print(module_1.assistant.id)