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
                    
                    
from skills.get_details_from_table import extract_data_from_cn
from typing import Annotated

@tool
def fetch_issue_description_from_casenumber(query: Annotated[str, "user's query about finding issue description"], case_number: Annotated[str, "input case number"]) -> str:
    """
    For a user's query about finding issue description for a given case number this tool returns the answer.
    Do not call this tool more than once.
    
    Args:
        query (string): User's question about finding issue description for a given case number.
        case_number (string): Input case number.

    Returns:
        issue description (string).
    """
    
    out_data = extract_data_from_cn(int(case_number))
    if type(out_data) == str:
        return f"No issue description found for case {case_number}"
    else:
        issue_subject, issue_desc = out_data
        return issue_subject + issue_desc