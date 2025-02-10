import io
import os
import uuid

from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobBlock, BlobClient, StandardBlobTier

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

if __name__ == '__main__':

    account_url = "https://dndblobdata.blob.core.windows.net"
    credential = DefaultAzureCredential()

    data_loader = DataLoader(account_url, credential)

    data_loader.create()

    container_name='grimms_tales'
    filepath = '/home/niclaswiegleb/projects/DnD-Homebrew-Generator/data_loaders/data/german_folk_tales/'
    filename = 'grimms_tale.xlsx'

    data_loader.upload_blob_file(container_name=container_name, filepath=filepath, filename=filename)
    