from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from langchain.docstore.document import Document
import pandas as pd
from datetime import datetime
import re
import json

llm_keyfile = "xxx.json"
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

creds_bq = service_account.Credentials.from_service_account_file(
                "xxx.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])

client_bq = bigquery.Client(credentials = creds_bq, project = "xxxxx")


def preProcess(msg):
    sent = msg.split("\n")

    out_sent = ""

    for i in range(len(sent)):
        if sent[i].strip().startswith('From:'):
            break
            
        if re.search("Google Cloud Support <EMAIL_ADDRESS>".lower(), sent[i]):
            break

        if re.search("CONFIDENTIALITY NOTICE:".lower(), sent[i]):
            break

        if re.search("Google Cloud Support, <EMAIL_ADDRESS>".lower(), sent[i]):
            break

        if re.search("The information contained in this transmission may contain privileged".lower(), sent[i]):
            break

        if re.search("This message contains information that is confidential".lower(), sent[i]):
            break

        out_sent += sent[i] + "\n"

    return out_sent


SYSTEM_PROMPT = """
# Role and Objective
You are an intelligent knowledge retrieval assistant. Your task is to analyze provided documents or URLs to extract the most relevant information for user queries.

# Instructions
1. Analyze the user's query carefully to identify key concepts and requirements.
2. Search through the provided sources for relevant information and output the relevant parts in the 'content' field.
3. If you cannot find the necessary information in the documents, return 'isIrrelevant: true', otherwise return 'isIrrelevant: false'.

# Constraints
- Do not make assumptions beyond available data
- Clearly indicate if relevant information is not found
- Maintain objectivity in source selection
"""


RAG_PROMPT = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response to the user's question, summarizing all information from the context data list, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.

---Question---

{question}

---Context---

{context_data}

---Answer---


"""


FMR_FORMAT = """
You are a helpful Support agent. Your task is to generate the first meaningful response (FMR) for the below Customer's issue.

Issue description:
```{issue}```

The FMR should begin with greating the Customer and acknowledge the issue. Rephrase the issue as you understood. Provide the solution and ask any
follow up question if necessary.

Below is a sample FMR template:

```Hi CUSTOMER_NAME ,

 
Thank you for contacting Google Cloud Platform Support. My name is SUPPORT_NAME and I will be assisting you with the issue today.

 
I understand that <rephrase what you have understood about the issue>

 
<add the investigation details or resolution details>

[1] <provide supporting links>

[1] <provide supporting links>

I hope the above information will be helpful for you. Please correct me if I have misunderstood any of your requirements/use case.

I will be waiting for your response.

Have a great day!


Thanks & Regards,
SUPPORT_NAME```

"""


from pydantic import BaseModel,Field
from typing import List
from langchain_core.output_parsers import StrOutputParser

class ResponseSchema(BaseModel):
    content: str = Field(...,description="The content of the document that is relevant or sufficient to answer the question asked")
    reasoning: str = Field(...,description="The reasoning for selecting The page with respect to the question asked")
    is_irrelevant: bool = Field(...,description="Specify 'True' if the content in the document is not sufficient or relevant to answer the question asked otherwise specify 'False' if the context or content is relevant to answer the question asked")


class RelevancySchemaMessage(BaseModel):
    source: ResponseSchema

relevancy_parser = StrOutputParser(pydantic_object=RelevancySchemaMessage)


def fetch_cases_data(cn):
    query = f"""
            select * from `table` where cn in unnest({cn}); 
    """
    
    res = client_bq.query(query).result().to_dataframe()
    
    return res


def extract_relevant_context(question,documents):    
    result = []
    for doc in documents:
        print(f"processing.. {doc.metadata['source']['cn']}")
        formatted_documents = f"Content : {doc.page_content}"
        system = f"{SYSTEM_PROMPT}\n\n# Available source\n\n{formatted_documents}"
        prompt = f"""Determine if the 'Avaiable source' content supplied is sufficient and relevant to ANSWER the QUESTION asked.
        QUESTION: {question}
        #INSTRUCTIONS TO FOLLOW
        1. Analyze the context provided thoroughly to check its relevancy to help formulizing a response for the QUESTION asked.
        2, STRICTLY PROVIDE THE RESPONSE IN A JSON STRUCTURE AS DESCRIBED BELOW:
            ```json
               {{"content":<<The content of the document that is relevant or sufficient to answer the question asked>>,
                 "reasoning":<<The reasoning for selecting The content with respect to the question asked>>,
                 "is_irrelevant":<<Specify 'True' if the content in the document is not sufficient or relevant. Specify 'False' if the content is sufficient to answer the QUESTION>>
                 }}
            ```
         """
        messages =[ {"role": "user", "parts": [{"text" : system}]},
                       {"role": "user", "parts": [{"text" : prompt}]},
                    ]
        response = generate(messages)
        #print(response)
        formatted_response = relevancy_parser.parse(response)
        formatted_response = formatted_response.lstrip("```json").rstrip("```")
        formatted_response = json.loads(formatted_response)
        result.append(formatted_response)
    final_context = []
    for items in result:
        if (items['is_irrelevant'] == False) or ( items['is_irrelevant'] == 'false') or (items['is_irrelevant'] == 'False'):
            final_context.append(items['content'])
    return final_context



def RAG(cn, question, mode = "FMR", **kwargs):
    additional_info = ""
    if ("additional_info" in kwargs.keys()):
        additional_info = "Additional Instructions: " + kwargs["additional_info"]
    similar_issues_data = fetch_cases_data(cn)
    #print(similar_issues_data)
    if len(similar_issues_data) == 0:
        return "Looks like table is empty, please check with developers.. :("

    docArray = []
    for i in range(len(similar_issues_data)):
        text = similar_issues_data.iloc[i]["desc"]
        doc =  Document(page_content = text)
        doc.metadata = {"source": {"cn" : ""}}
        doc.metadata["source"]["cn"] = str(similar_issues_data.iloc[i]["cn"])
        docArray.append(doc)

    final_context = extract_relevant_context(question, docArray)
    final_out = f"No such case is there in the table which concludes similar solution steps for the issue"
    if len(final_context) > 0:
        if mode == "FMR":
            question_prompt = FMR_FORMAT.format(issue = question)
            question_prompt = question_prompt + "\n" + additional_info
            prompt = RAG_PROMPT.format(question = question_prompt, context_data = final_context)
        else:
            prompt = RAG_PROMPT.format(question = question, context_data = final_context)
            prompt += additional_info
        final_out = generate(prompt)
        return final_out
    else:
        return final_out
