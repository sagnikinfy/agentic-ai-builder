from google.cloud import storage
from google.oauth2 import service_account
from typing import List, Union
import json
import os

project = "apigee-infosys"
key_file = "apigee.json"
bucket = "agentic-ai-infosys"

creds = service_account.Credentials.from_service_account_file(key_file)
storage_client = storage.Client(credentials = creds, project = project)
bucket = storage_client.get_bucket(bucket)

def check_if_exists(name: str, mode: str) -> List:
    """
    Check whether a skill/agents is exists in the bucket or not
    """
    return list(bucket.list_blobs(prefix=f"{mode}/{name}.py"))