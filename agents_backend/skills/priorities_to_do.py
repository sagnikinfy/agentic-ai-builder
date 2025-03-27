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
def fetch_gvc_data(ldap: str) -> str:
    creds_bq = service_account.Credentials.from_service_account_file(
                "infy_auto.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])

    client_bq = bigquery.Client(credentials = creds_bq, project = "upheld-caldron-411606")
    query = f"""
        select case_number, start_time,  gvc_link from `upheld-caldron-411606.callback.gcp_callback_status` 
        where owner_ldap = '{ldap}' and comment = '#tsr_confirmed' and status = 'open' and date(start_time) >= current_date() order by start_time; 
    """
    result = client_bq.query(query).result().to_dataframe()
    out = "You don't have any GVC today\n\n"
    if len(result) > 0:
        out = "You have below priority tasks:\n\nGVC - \n"
        for i in range(len(result)):
            out += f"{i+1}. For case #{result.iloc[i]['case_number']} Starts from {result.iloc[i]['start_time']}, GVC link - {result.iloc[i]['gvc_link']} .\n"
    
    return out

def fetch_negative_sentiment_cases(ldap: str) -> str:
    creds_bq = service_account.Credentials.from_service_account_file(
                "apigee.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])

    client = bigquery.Client(credentials = creds_bq, project = "apigee-infosys")
    query = f"""
        select CaseNumber, justification from `apigee-infosys.gcp_sentiment.RCA_Sentiment_Summary` where Sentiment_description = 'negative' 
        and Status = 'In Progress Google Support' limit 5; 
    """
    result = client.query(query).result().to_dataframe()
    out = "You don't have any high priority negative cases"
    if len(result) > 0:
        out = "High priority cases - \n"
        for i in range(len(result)):
            out += f"{i+1}. #{result.iloc[i]['CaseNumber']} - {result.iloc[i]['justification']}\n"
            
    return out

from typing import Literal, Annotated, Sequence
                    
                    
@tool
def priorities_to_do(query: Annotated[str, "user's query about priorities or urgent tasks to do"]) -> str:
    """
    Takes query about user's priorities or urgent tasks to do and returns the answer.
    Do not call this tool more than once.
    
    Args:
        query (string): User's question about priorities or urgent tasks to do.

    Returns:
        details about gvc and high priority cases which are user's priority tasks to do.
    """

    response = fetch_gvc_data("varanasim") + fetch_negative_sentiment_cases("varanasim")
    return response
    
    