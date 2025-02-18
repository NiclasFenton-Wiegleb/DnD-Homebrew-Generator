import json
import requests
import time

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from .base_module import BaseModule