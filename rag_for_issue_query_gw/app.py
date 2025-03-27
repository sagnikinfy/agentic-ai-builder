from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from typing import Union, List
from postgres import AsyncPGVectorDriver
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from google.cloud import bigquery
import ast
import json
from embd import embd_model
from google.oauth2 import service_account
from langchain.docstore.document import Document
from llm import generate
import tiktoken
from load_pdf import pdf_read
import nest_asyncio
nest_asyncio.apply()


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = ''
host = ""
driver = ""
user = ""
password = ""
db = ""


prompt = """---Role---

You are a helpful assistant. Generate a response to the user's question, summarizing all information in the given input context and incorporating any 
relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.

---context---

```{context_data}```

Output:
"""


def extract_data():
    path = ""
    data = pdf_read(path)
    #print(data[:40])
    def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
        encoder = tiktoken.encoding_for_model(model_name)
        tokens = encoder.encode(content)
        return tokens

    def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
        encoder = tiktoken.encoding_for_model(model_name)
        content = encoder.decode(tokens)
        return content
    
    tiktoken_model="gpt-4o"
    max_token_size = 12000
    overlap_token_size = 1000
    results = []
    tokens = encode_string_by_tiktoken(data, model_name=tiktoken_model)
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(tokens[start : start + max_token_size], model_name=tiktoken_model)
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )

    #print(results[0])
        
    docArray = []
    for i in results:
        doc =  Document(page_content = i["content"])
        doc.metadata = {"source": {"book" : "", "tokens" : "", "ldap" : ""}}
        doc.metadata["source"]["book"] = "GWS cases - results-20250304-172224.pdf"
        doc.metadata["source"]["tokens"] = i["tokens"]
        doc.metadata["source"]["chunk_order_index"] = i["chunk_order_index"]
        docArray.append(doc)

    return docArray
    

docArray = extract_data() 
#print(docArray)

@app.route("/isalive")
def is_alive():
    status_code = Response(status=200)
    return status_code


@app.route("/rag-query", methods=["POST"])
def predict():
    req_json = request.json
    #print(req_json)
    instance = req_json["query"]
    collection = req_json["coll"]
    param1 = req_json["keyword"]
    param2 = req_json["vector"]

    try:
        
        pginstance = AsyncPGVectorDriver(pg_conn = f"postgresql+asyncpg://{user}:{password}@{host}/{db}",
                                 embd_model = embd_model, collection_name = collection)

        vectorstore_retreiver = pginstance.as_retriever(search_kwargs={"k": 10})
        keyword_retriever = BM25Retriever.from_documents(docArray)
        keyword_retriever.k = 10
        ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver, keyword_retriever], weights=[param2, param1])
        docs = ensemble_retriever.invoke(instance)
        #print(docs)
        context = ""
        for i,j in enumerate(docs):
            context += j.page_content + "\n\n"
            
        system_prompt = prompt.format(context_data = context)
        #print(system_prompt)
        messages = []
        messages.append({"role": "user", "parts": [{"text" : system_prompt}]})
        messages.append({"role": "user", "parts": [{"text" : instance}]})
        #print(messages)
        out = generate(messages)

        return jsonify({
            "predictions": out
        })

    except Exception as e:

        return jsonify({
            "predictions": str(e)
        })



if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8000)
