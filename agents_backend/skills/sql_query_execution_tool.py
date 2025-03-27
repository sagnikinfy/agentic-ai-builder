from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm_keyfile = "infy_auto.json"
llm_project = "upheld-caldron-411606"
creds_llm = service_account.Credentials.from_service_account_file(llm_keyfile)

llm = ChatVertexAI(safety_settings=safety_settings, project = llm_project, 
                   credentials=creds_llm, location="us-central1",
                    model_name= 'gemini-1.5-flash-preview-0514',
                    temperature= 0.0,
                    top_p=0.8,
                    top_k=40,
                    verbose= True,
                    convert_system_message_to_human=False,
                    streaming = True, 
                    max_output_tokens = 8000)

from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Literal, Annotated, Sequence

@tool
def sql_query_execution_tool(sql: Annotated[str, "Input SQL query related to user's question about various metrics trends regarding 'escalations', 'CES', 'TSR', 'specialization', 'shard', 'customer', 'team' etc"]) -> str:
    """
    Takes a SQL query related to user's question about various metrics trends regarding 'escalations', 'CES', 'TSR', 'specialization', 'shard', 'customer', 'team' etc as input and executes it. Then returns the final answer as JSON format.
    Do not call this tool more than once.
    
    Args:
        sql (SQL query): A SQL query related to user's question about various metrics trends regarding 'escalations', 'CES', 'TSR', 'specialization', 'shard', 'customer', 'team' etc.
    
    Returns:
        Executed output: (JSON)
    
    """
    creds_bq = service_account.Credentials.from_service_account_file(
                    "apigee.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                                  "https://www.googleapis.com/auth/drive",
                                  "https://www.googleapis.com/auth/bigquery",])

    client = bigquery.Client(credentials = creds_bq, project = "apigee-infosys")
    
    
    try:
        try:
            sql = json.loads(sql)
            sql = sql["content"].split(":")[-1].strip()
        except Exception as e:
            sql = sql
        df_out = client.query(sql).result().to_dataframe()
        if df_out.isnull().values.any():
            df_out = df_out.fillna(0)
        return f"Executed output:\n {df_out.to_json()}"
    except Exception as e:
        return f"Executed output:\n {e}"