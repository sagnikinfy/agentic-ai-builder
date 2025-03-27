from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from typing import Union, List
from postgres import AsyncPGVectorDriver
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from google.cloud import bigquery
import ast
import json
from embd import Embeddings
from google.oauth2 import service_account
from langchain.docstore.document import Document
from llm import generate
import nest_asyncio
nest_asyncio.apply()


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret'
host = "10.128.0.33:5432"
driver = "asyncpg"
user = "postgres"
password = "mpostsagnik76542gres"
db = "embd"
collection = "collection_sim"
sum_key = "apigee.json"
sum_proj = "apigee-infosys"
embeddings = Embeddings()

creds_bq = service_account.Credentials.from_service_account_file(
                sum_key,scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])

'''
prompt = """---Role---

You are a helpful assistant. Generate a response to the user's question, 
summarizing all information in the given input context appropriate for the response length and format, and incorporating any 
relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.


---Goal---

You will be given a json data as context consisting integers as keys and issue/error description of a product or tech stack as values. 
Your task is to extract key values correspoding to top {cases_num} similar issue/error description with the same mentioned product or tech stack in the query.

    Searching process you must follow: 

    - First, understand error details and the product or tech stack from the given query error description.

    - Then, search the context json data, having similar error details with the exact same product (or tech stack) as mentioned in 
      the query error description. If any error code mentioned in the query, search for exact error code with for that product 
      or tech stack. 

      Example: You are given query as '502 error in gke'.
      Thinking process to find similar issue/error: Here the error code is '502', which is bad gateway server error, and the mentioned 
      product or tech stack is 'gke'. So you need to search for issue/error descriptions having this same exact error with 'gke' only.

    - Output will be the a list or array of top {cases_num} keys corresponding to similar issue/error description.  

    If there is no issue/error description which is similar to the given query, reply 'not found'.


---context---

```{context_data}```

Output:
"""
'''

prompt = """---Role---

You are a helpful assistant. Generate a response to the user's question, summarizing all information in the given input context and incorporating any
relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.


---Goal---

You will be given a json data as context consisting integers as keys and issue/error description of a product or tech stack as values.
Your task is to extract key values correspoding to top {cases_num} similar issue/error description with the same mentioned product or tech stack in the query.

    Searching process you must follow:

    - First, understand error details and the product or tech stack from the given query error description.

    - Then, search the context json data, having similar error details with the exact same product (or tech stack) as mentioned in
      the query error description. If any error code mentioned in the query, search for exact error code with for that product
      or tech stack.

      Example: You are given query as '502 error in gke'.
      Thinking process to find similar issue/error: Here the error code is '502', which is bad gateway server error, and the mentioned
      product or tech stack is 'gke'. So you need to search for issue/error descriptions having this same exact error with 'gke' only.

    - Output will be the a list or array of top {cases_num} keys corresponding to similar issue/error description.

    If there is no issue/error description which is similar to the given query, reply 'not found'.


---context---

```{context_data}```

Output:
"""


def extract_data():
    client = bigquery.Client(credentials = creds_bq, project = sum_proj)
    query = f"select * from `apigee-infosys.case_descriptions.issue_desc`"
    out = client.query(query).to_dataframe()
    docArray = []
    for i in range(len(out)):
        text = ""
        if (out.iloc[i]['tag'] == out.iloc[i]['details']):
            text = f"Issue description : {out.iloc[i]['tag']}"
        else:
            text = f"Issue description : {out.iloc[i]['tag']}\n{out.iloc[i]['details']}"
        doc =  Document(page_content = text)
        #doc.metadata = {"source": ""}
        #doc.metadata["source"] = str(out.iloc[i]["case_num"])
        doc.metadata = {"source": {"cn" : "", "prod" : "", "ldap" : ""}}
        doc.metadata["source"]["cn"] = str(out.iloc[i]["case_num"])
        doc.metadata["source"]["prod"] = str(out.iloc[i]["product"])
        doc.metadata["source"]["ldap"] = str(out.iloc[i]["ldap"])
        docArray.append(doc)
    return docArray

docArray = extract_data() 


@app.route("/isalive")
def is_alive():
    status_code = Response(status=200)
    return status_code


@app.route("/sim", methods=["POST"])
def predict():
    req_json = request.json
    instance = req_json["query"]
    thresh = req_json["thresh"]
    cnum = req_json["cnum"]
    param1 = req_json["keyword"]
    param2 = req_json["vector"]

    try:
        
        pginstance = AsyncPGVectorDriver(pg_conn = f"postgresql+asyncpg://{user}:{password}@{host}/{db}",
                                 embd_model = embeddings, collection_name = collection)

        vectorstore_retreiver = pginstance.as_retriever(search_kwargs={"k": 60})
        keyword_retriever = BM25Retriever.from_documents(docArray)
        keyword_retriever.k = 60
        ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver, keyword_retriever], weights=[param2, param1])
        docs = ensemble_retriever.invoke(instance)
        context_dict = {}
        metadata_dict = {}
        for i,j in enumerate(docs):
            context_dict[i+1] = j.page_content
            metadata_dict[i+1] = j.metadata["source"]
            
        system_prompt = prompt.format(cases_num = int(cnum), context_data = json.dumps(context_dict))
        messages = []
        messages.append({"role": "user", "parts": [{"text" : system_prompt}]})
        messages.append({"role": "user", "parts": [{"text" : instance}]})
        out = generate(messages)
        out_list = out.lstrip("```json").rstrip("```").strip()
        out_list = ast.literal_eval(out_list)
        output = [metadata_dict[i]["cn"] for i in out_list]

        return jsonify({
            "predictions": output
        })

    except Exception as e:

        return jsonify({
            "predictions": str(e)
        })



if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8000)
