from langchain.embeddings import VertexAIEmbeddings
from google.oauth2 import service_account
llm_keyfile = "xxx.json"
llm_project = "xxxxx"
creds_llm = service_account.Credentials.from_service_account_file(llm_keyfile)
embd_model = VertexAIEmbeddings(credentials = creds_llm, project = llm_project)
