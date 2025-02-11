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
        self.client = None

    def create(self):
        # Create the BlobServiceClient object    
        self.client = BlobServiceClient(self.account_url, credential=self.credentials)

    def upload_blob_file(self, container_name: str, filepath: str, filename: str):
        # upload a file to the named container or create the container if doesn't exist yet.
        try:
            container_client = self.client.get_container_client(container=container_name)

            with open(file=os.path.join(filepath, filename), mode='rb') as data:
                blob_client = container_client.upload_blob(name=filename, data=data, overwrite=True)
        
        except:
            container_client = self.client.create_container(container_name)

            with open(file=os.path.join(filepath, filename), mode='rb') as data:
                blob_client = container_client.upload_blob(name=filename, data=data, overwrite=True)

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
                    name="myOpenAI",
                    kind="azureOpenAI",
                    azure_open_ai_parameters=az_openai_parameter,
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

if __name__ == '__main__':

    account_url = "https://dndblobdata.blob.core.windows.net"
    credential = DefaultAzureCredential()

    data_loader = DataLoader(account_url, credential)

    data_loader.create()

    container_name='grimms_tales'
    filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    filename = 'grimms_tale.xlsx'

    data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)
    