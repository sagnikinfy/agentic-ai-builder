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
import asyncio
#import nest_asyncio
#nest_asyncio.apply()

sim_url_hybrid = "http://xxx"
llm_keyfile = "xxx.json"
llm_project = "xxxxx"
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
                    
                    
from skills.http_call import http_call
from typing import Annotated

@tool
def similar_case_search(query: Annotated[str, "user's query about finding similar cases"], issue_desc: Annotated[str, "Detailed description of an issue"]) -> str:
    """
    For a user's query about finding similar cases for an issue description returns a list of case numbers having similar issues.
    Do not call this tool more than once.
    
    Args:
        query (string): User's question about finding similar cases for an issue description.
        issue_desc (string): Input issue description.

    Returns:
        case numbers (string) having the similar issue.
    
    """
    
    payload = {"query": issue_desc, "cnum" : 5, "thresh" : 0.2, "keyword": 0.8, "vector": 0.2}
    out = asyncio.run(http_call(sim_url_hybrid, payload))
    if not isinstance(out, list):
        return f"There are no similar cases for this query issue: {query}"
    else:
        if (len(out) > 0):
            resp = "Similar case numbers are : "
            resp += ', '.join(str(i) for i in out)
            return resp
        else:
            return f"There are no similar cases for this query issue: {query}" 