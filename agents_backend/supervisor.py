from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
import vertexai
import json
from langchain_core.pydantic_v1 import BaseModel, Field

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

def supervisor(query: str):
    
    with open("agents/agents.json") as f:
        data = json.loads(f.read())
    worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description["prompt"]}' for member, description in data.items()])
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " crew of workers:\n"
        "Below are the workers details:\n\n"
        f"{worker_info}\n\n" 
        "Given the following user request,"
        " tell me which worker should act next ?\n"
        "Also, if you are not sure about assinging correct worker against the user request, or if you feel that there is no correct worker available to fulfill user's request, the just tell me 'FINISH'."
    )
    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH. and provide reasoning for the routing"""

        next: str = Field(description="Worker to route to next.")
        reasoning: str = Field(description="Support proper reasoning for routing to the worker")
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response.next
    return goto