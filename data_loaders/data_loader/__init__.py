import io
import os
import uuid

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobBlock, BlobClient, StandardBlobTier

from .data_loader import DataLoader