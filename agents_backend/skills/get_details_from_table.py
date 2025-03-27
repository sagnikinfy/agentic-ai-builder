from google.oauth2 import service_account
from google.cloud import bigquery
from typing import Union, Tuple

def extract_data_from_cn(cn: int) -> Union[Tuple[str, str], str]:
    creds_bq_topaz = service_account.Credentials.from_service_account_file(
                "topaz-poc-2024-new.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])
    client_topaz = bigquery.Client(credentials = creds_bq_topaz, project = "topaz-poc-2024")
    query = f"""
        select case_subject, concat(case_subject,"\\n", description) as issue from `topaz-poc-2024.agents_casedata.gcp_case_metadata` where case_number = @cn limit 1; 
    """
    query_params=[
        bigquery.ScalarQueryParameter("cn", "INTEGER", cn),
    ]

    job_config=bigquery.QueryJobConfig(
        query_parameters=query_params
    )

    out = client_topaz.query(query,job_config=job_config).result().to_dataframe()
    if len(out) > 0:
        return (out["case_subject"].values[0], out["issue"].values[0])
    else:
        return "issue description not found"