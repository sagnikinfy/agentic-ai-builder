import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account

llm_keyfile = "xx.json"
llm_project = "xxxxx"
creds_llm = service_account.Credentials.from_service_account_file(llm_keyfile)  
vertexai.init(project=llm_project, location="us-central1", credentials=creds_llm)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

def generate(msgs):
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )
        
    responses = model.generate_content(
      msgs,
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
    )
    
    r = ""
    for response in responses:
        r = r + response.text
    return r
