from google.cloud import storage
from google.oauth2 import service_account
import shutil
import os
import glob
from google.cloud import bigquery
import json

project = "apigee-infosys"
key_file = "apigee.json"
dataset = "metrics"
table = "load_job_history"
rca_bucket = "rca_rag"

creds = service_account.Credentials.from_service_account_file(key_file)
storage_client = storage.Client(credentials = creds, project = project)
bucket = storage_client.get_bucket(rca_bucket)
creds_bq = service_account.Credentials.from_service_account_file(
    key_file, scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/bigquery",
    ]
)

client_bq = bigquery.Client(credentials = creds_bq, project = project)

def insert_log(ldap, log, timestamp):
    query = f"""
        insert into `{project}.{dataset}.{table}` values(@ldap, @log, @timestamp)
    """
    query_params = [
        bigquery.ScalarQueryParameter("ldap", "STRING", ldap),
        bigquery.ScalarQueryParameter("log", "STRING", json.dumps(log)),
        bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", timestamp)
    ]

    job_config = bigquery.QueryJobConfig(query_parameters = query_params)
    query_job = client_bq.query(query, job_config = job_config)
    if not list(query_job.result()):
        return True
    else:
        return False

def check_if_exists(case_num):
    return list(bucket.list_blobs(prefix=f"adv_rag/{case_num}/"))

def check_if_exists_case_arr(case_num_arr):
    return [i for i in case_num_arr if not check_if_exists(i)]


def delete_local_folders(dir_name, cn):
    shutil.rmtree(f"{dir_name}/{cn}")



def upload_local_directory_to_gcs(local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file,gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)







    