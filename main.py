import sys
import os
import json
sys.path.append('./pipelines')
sys.path.append('./data_loaders')

import pandas as pd
from openai import AzureOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureAISearchConnection
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
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    ExhaustiveKnnAlgorithmConfiguration,
    HnswParameters,
    ExhaustiveKnnParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
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
                        api_key=AZUREs_OPENAI_API_KEY,
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
    SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
    AI_PROJECT = os.getenv("AI_PROJECT")
    AZURE_AI_SEARCH = os.getenv("AZURE_AI_SEARCH")
    AZURE_AI_SEARCH_TARGET = os.getenv("AZURE_AI_SEARCH_TARGET")
    AZURE_BLOB_DATA_SOURCE = os.getenv("AZURE_BLOB_DATA_SOURCE")

    # create a credentials
    credentials = ClientSecretCredential(
        client_id = client_id,
        client_secret = client_secret,
        tenant_id = tenant_id
    )

    # User-specified parameter
    USE_AAD_FOR_SEARCH = True  
    # index_name = "_____"
    vectorizer_name = "myOpenAI" if "openai" in index_name else None

    # Module variables
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    api_key = os.environ.get('AZURE_OPENAI_API_KEY')

    model="gpt-35-turbo" # replace with model deployment name.
    # DataLoader
    data_loader = DataLoader(account_url, credentials)
    
    # Set variable for what service to run
    options = ['create_embeddings', 'connect_vcstore', 'create_search_index'
                'create_skillset', 'run_indexer', 'create_blob', 'upload_embeddings_to_index']
    text = ', '.join(options)
    print('What would you like to do?')
    user_input = input(f'Available options:\n{text}\n')
    while user_input not in options:
        print('Invalid option.')
        user_input = input(f'Available options:\n{text}\n')

    if 'create_embeddings' == user_input:
        # Create embeddings from datasets
        filepath = input(f'Enter filepath: \n')
        directory = os.fsencode(filepath)
       
        input_data = []

        embeddings = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version="2024-03-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
            dimensions= 1500
        )
        
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith('.csv'):
                # Create dataframe from file and iterate over each line
                data = pd.read_csv(f'{filepath}/{filename}')

                for i in range(len(data.index)):
                    dict_data = {}
                    dict_data['id'] = str(i)
                    dict_data['filename'] = filename
                    chunk = []
                    for c in data.columns:
                        sentence = f'{c}: {data[c].iloc[i]}'
                        chunk.append(sentence)
                    chunks = '; '.join(chunk)
                    dict_data['line'] = chunks
                    vector = embeddings.embed_query(chunks)
                    dict_data['embedding'] = vector
                    input_data.append(dict_data)
            else:
                continue
            print(f'Completed processing: {filename}')
        # Output embeddings to docVectors.json file
        try:
            with open(f'{filepath}/docVectors.json', 'x') as f:
                json.dump(input_data, f)
        except:
            with open(f'{filepath}/docVectors.json', 'w') as f:
                json.dump(input_data, f)

    if 'upload_embeddings_to_index' == user_input:
        # Upload the created embeddings to Azure index
        filepath = input(f'Enter filepath: \n')
        index_name = input(f'Enter index name: \n')
        
        index_client = SearchIndexClient(endpoint= SEARCH_SERVICE_ENDPOINT,
                                        credential= AzureKeyCredential(ai_search_key))
        fields = [

                    SimpleField(name="id", type=SearchFieldDataType.String, 
                                key=True, sortable=True, 
                                filterable=True, facetable=True),
                    SearchableField(name="filename", type=SearchFieldDataType.String,
                                    filterable=True, facetable=True),
                    SearchableField(name="line", type=SearchFieldDataType.String),
                    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                searchable=True, vector_search_dimensions=1500, 
                                vector_search_profile_name="myHnswProfile")
                ]

        # Configure the vector search configuration  
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE
                    )
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myExhaustiveKnn",
                    kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="myExhaustiveKnn",
                )
            ]
        )
        semantic_config = SemanticConfiguration(
                            name="my-semantic-config",
                            prioritized_fields=SemanticPrioritizedFields(
                                content_fields=[SemanticField(field_name="line")],
                                keywords_fields=[SemanticField(field_name="filename")]
                            )
                        )
        # Create the semantic settings with the configuration
        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create the search index with the semantic settings
        index = SearchIndex(name=index_name, fields=fields,
                            vector_search=vector_search, 
                            semantic_search=semantic_search)
        result = index_client.create_or_update_index(index)
        print(f'{result.name} created')

        # Upload some documents to the index
        with open(filepath, 'r') as file:  
            documents = json.load(file)
        
        search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT,
                                    index_name=index_name,
                                    credential=AzureKeyCredential(ai_search_key))
        result = search_client.upload_documents(documents=documents)
        print(f"Uploaded {len(documents)} documents") 


    if 'connect_vcstore' == user_input:
        # Create and connect vector store to base module
        # index_name = input(f'Enter index name: \n')
        filepath = input(f'Enter filepath: \n')
        # query = input(f'Query: \n')
        # model = SentenceTransformer(AZURE_OPENAI_MODEL_NAME)
        # query_vector = model.encode([query])[0]
        # search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT,
        #                             index_name=index_name,
        #                             credential=AzureKeyCredential(ai_search_key))
        # vector_query = VectorizedQuery(vector=query_vector, 
        #                        k_nearest_neighbors=3, 
        #                        fields="embedding")
        # # Connect to Azure AI Foundry project and AI search service resource
        # wps_connection = AzureAISearchConnection(
        #         name=AZURE_AI_SEARCH,
        #         endpoint=AZURE_AI_SEARCH_TARGET,
        #         credentials=credentials,
        #     )
        # ml_client = MLClient(credentials, SUBSCRIPTION_ID, RESOURCE_GROUP, AI_PROJECT)
        # # ml_client.connections.create_or_update(wps_connection)
        # ai_search_connection = ml_client.connections.get(AZURE_AI_SEARCH)

        # index_langchain_retriever = get_langchain_retriever_from_index(ai_search_connection.path)
        embeddings = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version="2024-03-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME
        )
        
        vector_store = AzureSearch(
            azure_search_endpoint= ai_search_endpoint,
            azure_search_key=ai_search_key,
            index_name=index_name,
            embedding_function=embeddings.embed_query
        )

        docs = []

        # for file in os.listdir(directory):
        #     filename = os.fsdecode(file)
        #     if filename.endswith('.csv'):
        loader = CSVLoader(filepath,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                })
        docs_lazy = await loader.lazy_load()
        for doc in docs_lazy:
            docs.append(doc)
        print(docs)
        # docs = vector_store.similarity_search(
        #     query=query,
        #     k=3,
        #     search_type='similarity'
        # )
        # print(docs[0].page_content)
        # retriever = vector_store.as_retriever()
        # # Initialise LLM
        # llm = AzureChatOpenAI(
        #     openai_api_version="2024-06-01",
        #     api_key=AZURE_OPENAI_API_KEY,
        #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
        #     azure_deployment=AZURE_OPENAI_MODEL_NAME, # verify the model name and deployment name
        #     temperature=0.8,
        # )

        # # Create RAG context model
        # contextualize_q_system_prompt = """Given a chat history and the latest user question \
        # which might reference context in the chat history, formulate a standalone question \
        # which can be understood without the chat history. Do NOT answer the question, \
        # just reformulate it if needed and otherwise return it as is."""

        # contextualize_q_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", contextualize_q_system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human", "{input}"),
        #     ]
        # )
        # history_aware_retriever = create_history_aware_retriever(
        #     llm, retriever, contextualize_q_prompt
        # )
        # # Create chat
        # qa_system_prompt = """You are an assistant for question-answering tasks. \
        # Use the following pieces of retrieved context to answer the question. \
        # If you don't know the answer, just say that you don't know. \
        # Use three sentences maximum and keep the answer concise.\

        # {context}"""

        # qa_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", qa_system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human", "{input}"),
        #     ]
        # )

        # question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # query = input(f'Query: \n')
        # print(rag_chain.invoke({'input': query}))
    
    if 'create_search_index' == user_input:
        # Create search index
        create_search_index(
            data_loader,
            USE_AAD_FOR_SEARCH,
            index_name, vectorizer_name,
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
            AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_NAME,
            ai_search_key,
            SEARCH_SERVICE_ENDPOINT)

    if 'create_skillset' == user_input:
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
    if 'run_indexer' == user_input:
        # Run Indexer
        data_source = AZURE_BLOB_DATA_SOURCE

        skillset_name = f"vector-1741055615730-skillset"
        data_loader.create_indexer_client(SEARCH_SERVICE_ENDPOINT, credential=AzureKeyCredential(ai_search_key))
        data_loader.create_and_run_indexer(
            index_name='vector-1741055615730',
            skillset_name=skillset_name,
            data_source='vector-1741055615730-datasource',
            endpoint=SEARCH_SERVICE_ENDPOINT,
            credential=AzureKeyCredential(ai_search_key))

    if 'create_blob' == user_input:
        # Create Blob Storage
        data_loader.create_blob()

        container_name='dnd-generator-context-data'
        filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/dnd'
        filename = 'spells.csv'

        data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)

        # Module 1 execution
        module_1 = Module1()
        module_1.load_prompt()
        module_1.create(endpoint= endpoint, api_key= api_key, model= model)
        print(module_1.assistant.id)