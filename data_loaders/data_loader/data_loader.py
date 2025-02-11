import io
import os
import uuid
import time

import cohere
import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobBlock, BlobClient, StandardBlobTier
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureMachineLearningSkill,
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIModelName,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    ScalarQuantizationCompression,
    ScalarQuantizationParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SplitSkill,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import (
    HybridCountAndFacetMode,
    HybridSearch,
    SearchScoreThreshold,
    VectorSimilarityThreshold,
    VectorizableTextQuery,
    VectorizedQuery
)


class DataLoader():

    def __init__(self, account_url, credentials):
        self.account_url = account_url
        self.credentials = credentials
        self.blob_client = None
        self.index_client = None
        self.indexer_client = None

    def create_blob(self):
        # Create the BlobServiceClient object    
        self.bob_client = BlobServiceClient(self.account_url, credential=self.credentials)

    def upload_blob_file(self, container_name: str, filepath: str, filename: str):
        # upload a file to the named container or create the container if doesn't exist yet.
        try:
            container_client = self.bob_client.get_container_client(container=container_name)

            with open(file=os.path.join(filepath, filename), mode='rb') as data:
                blob_client_file = container_client.upload_blob(name=filename, data=data, overwrite=True)
        
        except:
            container_client = self.bob_client.create_container(container_name)

            with open(file=os.path.join(filepath, filename), mode='rb') as data:
                blob_client_file = container_client.upload_blob(name=filename, data=data, overwrite=True)

    def authenticate_azure_search(self, api_key=None, use_aad_for_search=False):
        #Allows user to choose whether to use credentials or api-key to use for authentication
        if use_aad_for_search:
            print("Using AAD for authentication.")
            credential = self.credentials
        else:
            print("Using API keys for authentication.")
            if api_key is None:
                raise ValueError("API key must be provided if not using AAD for authentication.")
            credential = AzureKeyCredential(api_key)
        
        return credential

    def create_fields(self, vector_search_dimensions):
        # Create fields used for initialising index search
        return [
            SearchField(
                name="parent_id",
                type=SearchFieldDataType.String,
                sortable=True,
                filterable=True,
                facetable=True,
            ),
            SearchField(name="title", type=SearchFieldDataType.String),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
                analyzer_name="keyword",
            ),
            SearchField(name="chunk", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=vector_search_dimensions,
                vector_search_profile_name="myHnswProfileSQ",
                stored=False
            ),
        ]
        
    def create_vector_search_configuration(self, vectorizer_name, az_openai_parameter):
        # Create the configurations for a vector search
        return VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hsnw-001",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="exhaustiveknn-001",
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    ),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfileSQ",
                    algorithm_configuration_name="hsnw-001",
                    compression_configuration_name="myScalarQuantization",
                    vectorizer=vectorizer_name,
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="exhaustiveknn-001",
                    vectorizer=vectorizer_name,
                ),
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    parameters=az_openai_parameter,
                    vectorizer_name=vectorizer_name
                ),
            ],
            compressions=[
                ScalarQuantizationCompression(
                    compression_name="myScalarQuantization",
                    rerank_with_original_vectors=True,
                    default_oversampling=10,
                    parameters=ScalarQuantizationParameters(quantized_data_type="int8"),
                )
            ],
        )
    
    def create_semantic_search_configuration(self):
        # Search configuration for semantic search
        return SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="mySemanticConfig",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="chunk")]
                    ),
                )
            ]
        )
    
    def create_search_index(self, index_name, fields, vector_search, semantic_search, search_service_endpoint, credential=None):
        # Takes the various configurations and creates a search index
        if credential == None:
            index_client = SearchIndexClient(
                endpoint=search_service_endpoint, credential=self.credentials
            )
        else:
            index_client = SearchIndexClient(
                endpoint=search_service_endpoint, credential=credential
            )
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        self.index_client = index_client.create_or_update_index(index)

    def create_split_skill(self):
        """Creates a split skill to chunk documents into pages."""
        return SplitSkill(
            description="Split skill to chunk documents",
            text_split_mode="pages",
            context="/document",
            maximum_page_length=2000,
            page_overlap_length=500,
            inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
            outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
        )   

    def create_embedding_skill_openai(self, azure_openai_endpoint, azure_openai_embedding_deployment, azure_openai_key):
        """Defines the embedding skill for generating embeddings via Azure OpenAI."""
        return AzureOpenAIEmbeddingSkill(
            description="Skill to generate embeddings via Azure OpenAI",
            context="/document/pages/*",
            resource_url=azure_openai_endpoint,
            deployment_name=azure_openai_embedding_deployment,
            api_key=azure_openai_key,
            model_name=AzureOpenAIModelName.TEXT_EMBEDDING3_LARGE,
            dimensions=3072, # Take advantage of the larger model with variable dimension sizes
            inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
            outputs=[OutputFieldMappingEntry(name="embedding", target_name="vector")],
        )

    def create_index_projections(self, index_name):
        """Creates index projections for use in a skillset."""
        vector_source = ("/document/pages/*/vector")
        return SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                        InputFieldMappingEntry(name="vector", source=vector_source),
                        InputFieldMappingEntry(
                            name="title", source="/document/metadata_storage_name"
                        ),
                    ],
                ),
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

    def create_indexer_client(self, search_service_endpoint, credential):
        # Creates indexer client object
        self.indexer_client = SearchIndexerClient(
                search_service_endpoint, credential=credential
            )

    def create_skillset(self, skillset_name, skills, index_projections):
        """Creates or updates the skillset with embedding and indexing projection skills."""
        if self.indexer_client != None:
            client = self.indexer_client
            skillset = SearchIndexerSkillset(
                name=skillset_name,
                description="Skillset to chunk documents and generate embeddings",
                skills=skills,
                index_projections=index_projections,
            )
            try:
                client.create_or_update_skillset(skillset)
                print(f"Skillset '{skillset_name}' created or updated.")
            except Exception as e:
                print(f"Failed to create or update skillset '{skillset_name}': {e}")
        else:
            raise ValueError('Client object cannot be None. Create client first using create_indexer_client method.')

    def create_and_run_indexer(self, index_name, skillset_name, data_source, endpoint, credential):
        # Creates and runs an indexer over a provided data_source.
        indexer_name = f"{index_name}-indexer"
        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer to index documents and generate embeddings",
            skillset_name=skillset_name,
            target_index_name=index_name,
            data_source_name=data_source,
            field_mappings=[FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")],
        )
        if self.indexer_client != None:
            client = self.indexer_client
        else:
            raise ValueError('Client object cannot be None. Create client first using create_indexer_client method.')
        client.create_or_update_indexer(indexer)
        client.run_indexer(indexer_name)
        print(f"{indexer_name} is created and running.")

if __name__ == '__main__':

    account_url = "https://dndblobdata.blob.core.windows.net"
    credential = DefaultAzureCredential()

    data_loader = DataLoader(account_url, credential)

    data_loader.create()

    container_name='grimms_tales'
    filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    filename = 'grimms_tale.xlsx'

    data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)
    