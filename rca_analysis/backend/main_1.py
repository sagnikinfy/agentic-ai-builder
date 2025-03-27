from dataclasses import dataclass, field, asdict
from typing import TypedDict, Union, Literal, Generic, TypeVar,Any, Union, cast, Type, List, Tuple
import numpy as np
import asyncio
from datetime import datetime
import os
from hashlib import md5
import tiktoken
from langchain.embeddings import VertexAIEmbeddings
from google.oauth2 import service_account
import re
import networkx as nx
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from collections import Counter, defaultdict
import html
import json
import logging
from functools import partial,wraps
import pandas as pd
import datetime
from nanovec_storage_reader import NanoVectorDB
from google.cloud import bigquery
from google.cloud import storage
import base64
logger = logging.getLogger("rag")
loop = asyncio.new_event_loop()


llm_keyfile = "infy_auto.json"
llm_project = "upheld-caldron-411606"
creds_llm = service_account.Credentials.from_service_account_file(llm_keyfile)
embd_model = VertexAIEmbeddings(credentials = creds_llm, project = llm_project)   
vertexai.init(project=llm_project, location="us-central1", credentials=creds_llm)
creds_bq = service_account.Credentials.from_service_account_file(
                "apigee.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])
client = storage.Client(credentials = creds_bq, project = "apigee-infosys")


def read_from_storage(bucket: str, file: str) -> Union[str, None]:
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(file)
    if blob is None:
        return None
    else:
        return blob.download_as_string()
    

def write_to_storage(bucket: str, path: str, name_space: str, data: dict):
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(f"{path}/vdb_{name_space}.json") 
    with blob.open(mode='w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)
    
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "local"
    only_need_context: bool = True
    only_need_prompt: bool = True
    response_type: str = "Multiple Paragraphs"
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
    

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")

## storage base class

@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass
    

@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError
        

@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError
        

@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used.")



## Embedding function

def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


@wrap_embedding_func_with_attrs(embedding_dim=768)
async def embedding(texts: list[str], batch: int) -> np.ndarray:
    embeddings = await loop.run_in_executor(None, embd_model.embed_documents, texts, batch)
    return np.array([dp for dp in embeddings])


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    encoder = tiktoken.encoding_for_model(model_name)
    tokens = encoder.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    encoder = tiktoken.encoding_for_model(model_name)
    content = encoder.decode(tokens)
    return content



## Vector db and Graph db

@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        self._bucket = self.global_config["working_dir"]
        self._data = read_from_storage(self._bucket, f"{self.global_config['storage_path']}/kv_store_{self.namespace}.json")
        if self._data is not None:
            self._data = json.loads(self._data)
            logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_to_storage(self._bucket, self.global_config['storage_path'], self.namespace, self._data)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}
        
        

@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._bucket = self.global_config["working_dir"]
        self._client_file_name =  f"{self.global_config['storage_path']}/vdb_{self.namespace}.json"
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(self.embedding_func.embedding_dim, bucket =self._bucket, file = self._client_file_name)
        self.cosine_better_than_threshold = self.global_config.get("cosine_better_than_threshold", self.cosine_better_than_threshold)


    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query], 1)
        embedding = embedding[0]
        results = self._client.query(query=embedding,top_k=top_k,better_than_threshold=self.cosine_better_than_threshold)
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

        
        
@dataclass
class NetworkXStorage(BaseGraphStorage):
    
    @staticmethod
    def load_nx_graph(data) -> nx.Graph:
        try:
            reader = nx.GraphMLReader()
            header = b'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
            new_bytes = data.replace(b"<graphml>", header)
            glist = list(reader(string=new_bytes))
            if len(glist) == 0:
                raise nx.NetworkXError("file not successfully read as graphml")
            return glist[0]
        except Exception as e:
            print("exception occurred while reading the graphml")
            return None

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._bucket = self.global_config["working_dir"]
        self._graphml_xml_file = read_from_storage(self._bucket, f"{self.global_config['storage_path']}/graph_{self.namespace}.graphml")
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._bucket}/{self.global_config['storage_path']}/graph_{self.namespace}.graphml with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids




GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "role", "technology", "organization", "event", "location", "concept"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
Example 4:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
This is about the tag CASE_COMMENTS. This tag was created at 2024-10-11 11:14:50-07. This tag was created by Internal. Below is the details related to it:
Hello,

Thank you for contacting Google Cloud Support.

This message is to confirm that we've received your quota request for project 'netapp-na-ne2-icx'. Quota increase requests typically take two business days to process. If this quota request is very urgent, please respond to this message so that our support agents can have full context when processing your quota increase request.

While we make every effort to provide you with a resolution to your case within two business days, please understand that some of the quota increase requests may require further evaluation which takes additional time.

If you have any further information and clarification you would like to include in your request, please feel free to reply to this message.

Best regards,
Google Cloud Support
#############
("entity"{tuple_delimiter}"tag"{tuple_delimiter}"role"{tuple_delimiter}"tag refers to the event which is here 'CASE_COMMENTS'. There are different tags based on given events."){record_delimiter}
("entity"{tuple_delimiter}"CASE_COMMENTS"{tuple_delimiter}"event"{tuple_delimiter}"CASE_COMMENTS here refers to an entity that means what the subject is all about"){record_delimiter}
("entity"{tuple_delimiter}"Creation time"{tuple_delimiter}"event"{tuple_delimiter}"Creation time is when the event 'CASE_COMMENTS' was created."){record_delimiter}
("entity"{tuple_delimiter}"Internal"{tuple_delimiter}"person"{tuple_delimiter}"Internal refers to the sender or creator of the event."){record_delimiter}
("entity"{tuple_delimiter}"Google Cloud"{tuple_delimiter}"technology"{tuple_delimiter}"Users use services provided by Google Cloud which is the central part of this topic."){record_delimiter}
("entity"{tuple_delimiter}"Google Cloud Support"{tuple_delimiter}"organization"{tuple_delimiter}"Google Cloud Support refers to a technical team specialized in Google Cloud technology who help to resolve technical issues of the users."){record_delimiter}
("entity"{tuple_delimiter}"project"{tuple_delimiter}"organization"{tuple_delimiter}"project 'netapp-na-ne2-icx' is a container that organizes Google Cloud resources for the users."){record_delimiter}
("entity"{tuple_delimiter}"agents"{tuple_delimiter}"person"{tuple_delimiter}"agents here refers to the individuals who works for Google Cloud Support and communicates with users to resolve technical issues."){record_delimiter}
("entity"{tuple_delimiter}"quota increase"{tuple_delimiter}"event"{tuple_delimiter}"quota increase is a request to increase the amount of a shared resource in Google Cloud."){record_delimiter}
("relationship"{tuple_delimiter}"tag"{tuple_delimiter}"CASE_COMMENTS"{tuple_delimiter}"From the description it is mentioned that tag of the event is CASE_COMMENTS."{tuple_delimiter}"notation, tag, topic type"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"CASE_COMMENTS"{tuple_delimiter}"Creation time"{tuple_delimiter}"Creation time is when the event 'CASE_COMMENTS' was created."{tuple_delimiter}"event time"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"CASE_COMMENTS"{tuple_delimiter}"Internal"{tuple_delimiter}"Internal sender created the event 'CASE_COMMENTS'."{tuple_delimiter}"action, creation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Google Cloud"{tuple_delimiter}"Google Cloud Support"{tuple_delimiter}"Google Cloud Support works to address user's issues working with Google Cloud technology."{tuple_delimiter}"collective action, technical support"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Google Cloud"{tuple_delimiter}"project"{tuple_delimiter}"Google Cloud has its project where users work with various resources."{tuple_delimiter}"resource, products"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"agent"{tuple_delimiter}"Google Cloud Support"{tuple_delimiter}"agents work for Google Cloud Support."{tuple_delimiter}"support, resource"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"agents"{tuple_delimiter}"quota increase"{tuple_delimiter}"From the description it is mentioned here that the agents process the user's request about quota increase."{tuple_delimiter}"resource, process, products"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"notation, tag, topic type, event time, resource, products, support, process"){completion_delimiter}
#############################
Example 5:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
On 2024-10-21 08:42:21-07 for case number #54341396: Escalation (#/54342835) has been Closed@Fedor Protkov
The case status of the case number #54341396 was changed to WOCR from IPGS on 2024-10-20 10:28:53-07
A bug (details : 374720404) related to the case number #54341396 was created on 2024-10-21 06:04:38.129-07
#############
("entity"{tuple_delimiter}"case number"{tuple_delimiter}"role"{tuple_delimiter}"case number refers here to a issue ticket number. case number can be different for different issues."){record_delimiter}
("entity"{tuple_delimiter}"54341396"{tuple_delimiter}"concept"{tuple_delimiter}"54342835 refers here to the unique case or ticket number."){record_delimiter}
("entity"{tuple_delimiter}"Escalation"{tuple_delimiter}"event"{tuple_delimiter}"Escalation here refers to an entity that explains an issue event."){record_delimiter}
("entity"{tuple_delimiter}"54342835"{tuple_delimiter}"concept"{tuple_delimiter}"54341396 refers here to the unique escalation id for the case with number '54341396'."){record_delimiter}
("entity"{tuple_delimiter}"@Fedor Protkov"{tuple_delimiter}"person"{tuple_delimiter}"@Fedor Protkov refers here to a person who is working on this case."){record_delimiter}
("entity"{tuple_delimiter}"WOCR"{tuple_delimiter}"event"{tuple_delimiter}"WOCR here refers to an entity that explains an event. It defines a status of the case."){record_delimiter}
("entity"{tuple_delimiter}"IPGS"{tuple_delimiter}"event"{tuple_delimiter}"IPGS here refers to an entity that explains an event. It defines a status of the case."){record_delimiter}
("entity"{tuple_delimiter}"BUG"{tuple_delimiter}"event"{tuple_delimiter}"BUG here refers to an entity that explains an issue event"){record_delimiter}
("entity"{tuple_delimiter}"374720404"{tuple_delimiter}"concept"{tuple_delimiter}"374720404 refers here to the unique BUG id for this case with number '54341396'."){record_delimiter}
("relationship"{tuple_delimiter}"case number"{tuple_delimiter}"54341396"{tuple_delimiter}"54341396 defines the unique case number of this case."{tuple_delimiter}"resource, notation, tag"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"54341396"{tuple_delimiter}"Escalation"{tuple_delimiter}"From the description, case number 54341396 had an Escalation which got closed on 2024-10-21 08:42:21-07."{tuple_delimiter}"event type, event time, issue"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Escalation"{tuple_delimiter}"54342835"{tuple_delimiter}"54342835 defines the unique id of the Escalation for this case."{tuple_delimiter}"resource, notation, tag"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"@Fedor Protkov"{tuple_delimiter}"Escalation"{tuple_delimiter}"@Fedor Protkov is an agent who worked on this case, closed the escalation on  2024-10-21 08:42:21-07."{tuple_delimiter}"action, technical support"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"WOCR"{tuple_delimiter}"IPGS"{tuple_delimiter}"The case status of this case #54341396 was in IPGS. After that it got changed to WOCR on 2024-10-20 10:28:53-07."{tuple_delimiter}"event, action, creation, status"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"54341396"{tuple_delimiter}"BUG"{tuple_delimiter}"From the description, case number 54341396 had a BUG which was created on 2024-10-21 06:04:38.129-07."{tuple_delimiter}"event type, event time, issue"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"BUG"{tuple_delimiter}"374720404"{tuple_delimiter}"374720404 defines the unique id of the BUG for this case."{tuple_delimiter}"resource, notation, tag"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"notation, tag, event time, event type, issue, action, technical support, resource, creation, status"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, considering the previous chat history, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Previous chat history---

{chat_history}

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""



## Entity Relationship extraction helper menthods 


def content_messages(*args: str):
    roles = ["user", "model"]
    return [
        {"role": roles[i % 2], "parts": [{"text" : content}]} for i, content in enumerate(args)
    ]

def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def clean_str(input: Any) -> str:
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

async def _handle_single_entity_extraction(record_attributes: list[str],chunk_key: str):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_single_relationship_extraction(record_attributes: list[str],chunk_key: str):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _handle_entity_relation_summary(entity_or_relation_name: str,description: str,global_config: dict) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.info(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

async def _merge_nodes_then_upsert(entity_name: str,nodes_data: list[dict],knowledge_graph_inst: BaseGraphStorage,
                                   global_config: dict):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(src_id: str, tgt_id: str, edges_data: list[dict], knowledge_graph_inst: BaseGraphStorage,
                                   global_config: dict):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(chunks: dict[str, TextChunkSchema],knowledge_graph_inst: BaseGraphStorage,
                           entity_vdb: BaseVectorStorage,relationships_vdb: 
                           BaseVectorStorage,global_config: dict) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0
    
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        #print(hint_prompt)
        #print("\n\n\n\n")
        final_result = await use_llm_func(hint_prompt)
        history = content_messages(hint_prompt, final_result)
        #print(content)
    
        
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += content_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        #print(final_result)
        #print("\n")
                
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"""{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), 
            {already_relations} relations(duplicated)\r""",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        print("Didn't extract any entities.")
        return None
    if not len(all_relationships_data):
        print(
            "Didn't extract any relationships."
        )
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
        
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst



## LLM helper methods


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


def generate(model, msgs, max_toks):
    #print("llm call")
    model = GenerativeModel(
        model,
    )
    if not max_toks:
        responses = model.generate_content(
          msgs,
          generation_config=generation_config,
          safety_settings=safety_settings,
          stream=True,
        )
        
    else:
        config = {
            "max_output_tokens": int(max_toks),
            "temperature": 1,
            "top_p": 0.95,
        }
        
        responses = model.generate_content(
          msgs,
          generation_config=config,
          safety_settings=safety_settings,
          stream=True,
        )
    
    r = ""
    for response in responses:
        r = r + response.text
    return r
    


def limit_async_func_call(max_size: int, waiting_time: float = 0.0001):

    def final_decro(func):
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waiting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


async def llm_if_cache(model, prompt, system_prompt=None, history_messages=[],**kwargs) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "user", "parts": [{"text" : system_prompt}]})
    messages.extend(history_messages)
    messages.append({"role": "user", "parts": [{"text" : prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
        
    max_toks = kwargs.get("max_tokens", None)
    response = await loop.run_in_executor(None, generate, model, messages, max_toks)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response, "model": model}}
        )
    return response


async def llm_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await llm_if_cache(
        "gemini-1.5-pro-001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )




## Retrieval helper methods


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

def list_of_list_to_csv(data: list[list]):
    return "\n".join([",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data])


async def _find_most_related_text_unit_from_entities(node_datas: list[dict],query_param: QueryParam,
                                                     text_chunks_db: BaseKVStorage[TextChunkSchema],
                                                     knowledge_graph_inst: BaseGraphStorage):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(node_datas: list[dict], query_param: QueryParam,
                                                 knowledge_graph_inst: BaseGraphStorage):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _find_most_related_entities_from_relationships(edge_datas: list[dict], query_param: QueryParam,
                                                         knowledge_graph_inst: BaseGraphStorage):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas

async def _find_related_text_unit_from_relationships(edge_datas: list[dict], query_param: QueryParam,
                                                     text_chunks_db: BaseKVStorage[TextChunkSchema],
                                                     knowledge_graph_inst: BaseGraphStorage):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def _build_local_query_context(query,knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage,
                                     text_chunks_db: BaseKVStorage[TextChunkSchema],
                                     query_param: QueryParam) -> Tuple[str, list]:
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    context_chunks = [i["content"] for i in use_text_units]
    return (f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
""", context_chunks)
'''
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
'''


async def _build_global_query_context(keywords, knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage,
                                      relationships_vdb: BaseVectorStorage,
                                      text_chunks_db: BaseKVStorage[TextChunkSchema],
                                      query_param: QueryParam) -> Tuple[str, list]:
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    context_chunks = [i["content"] for i in use_text_units]
    return (f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
""", context_chunks)

'''
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"] + "</St>"])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
'''


def combine_contexts(high_level_context, low_level_context):
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    h_ent = list_of_list_to_csv(entites_section_list)
    h_rel = list_of_list_to_csv(relations_section_list)
    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        
        return entities, relationships
    
    if high_level_context[0] is None:
        logger.warning(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships = "", ""
    else:
        hl_entities, hl_relationships = extract_sections(high_level_context[0])
        
    if low_level_context is None:
        logger.warning(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships =  "", ""
    else:
        ll_entities, ll_relationships = extract_sections(low_level_context[0])
        
    combined_entities_set = set(
        filter(None, hl_entities.strip().split("\n") + ll_entities.strip().split("\n"))
    )
    combined_entities = "\n".join(combined_entities_set)
    combined_entities = combined_entities.replace(h_ent + "\n", "")
    combined_entities = h_ent + "\n" + combined_entities

    combined_relationships_set = set(
        filter(
            None,
            hl_relationships.strip().split("\n") + ll_relationships.strip().split("\n"),
        )
    )
    combined_relationships = "\n".join(combined_relationships_set)
    combined_relationships = combined_relationships.replace(h_rel + "\n", "")
    combined_relationships = h_rel + "\n" + combined_relationships
    
    
    combined_sources_set = set(
        filter(None, high_level_context[1] + low_level_context[1])
    )
    
    combined_sources = [["id", "content"]]
    counter = 0
    for i in combined_sources_set:
        combined_sources.append([counter, i])
        counter += 1
        
    #combined_sources = "\n".join(combined_sources_set)
    combined_sources = list_of_list_to_csv(combined_sources)
    
    return f"""
-----Entities-----
```csv
{combined_entities}
-----Relationships-----
{combined_relationships}
-----Sources-----
{combined_sources}
"""


    
    

def combine_contexts_v1(high_level_context, low_level_context):
    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources
    
    if high_level_context is None:
        logger.warning(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        logger.warning(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    combined_entities_set = set(
        filter(None, hl_entities.strip().split("\n") + ll_entities.strip().split("\n"))
    )
    combined_entities = "\n".join(combined_entities_set)

    combined_relationships_set = set(
        filter(
            None,
            hl_relationships.strip().split("\n") + ll_relationships.strip().split("\n"),
        )
    )
    combined_relationships = "\n".join(combined_relationships_set)
    
    combined_sources_set = set(
        filter(None, hl_sources.strip().split("</St>") + ll_sources.strip().split("</St>"))
    )
    combined_sources = "\n".join(combined_sources_set)

    return f"""
-----Entities-----
```csv
{combined_entities}
-----Relationships-----
{combined_relationships}
-----Sources-----
{combined_sources}
"""


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


async def local_query(query, history: str, knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage,
                      relationships_vdb: BaseVectorStorage,
                      text_chunks_db: BaseKVStorage[TextChunkSchema], query_param: QueryParam, global_config: dict) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, chat_history=history, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def hybrid_query(query, history: str, knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage,
                       relationships_vdb: BaseVectorStorage,
                       text_chunks_db: BaseKVStorage[TextChunkSchema],
                       query_param: QueryParam,
                       global_config: dict,) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    try:
        keywords_data = json.loads(result)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        
    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
        
    #print(low_level_context)
    #print(high_level_context)

    context = combine_contexts(high_level_context, low_level_context)
    #print(context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, chat_history=history, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


async def naive_query(query: str, history: str, chunks_vdb: BaseVectorStorage, 
                      text_chunks_db: BaseKVStorage[TextChunkSchema],query_param: QueryParam,global_config: dict):
    
    def context_reorder(documents: List[str]) -> List[str]:
        documents.reverse()
        reordered_result = []
        for i, value in enumerate(documents):
            if i % 2 == 1:
                reordered_result.append(value)
            else:
                reordered_result.insert(0, value)
        return reordered_result
    
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    maybe_trun_chunks = [c["content"] for c in maybe_trun_chunks]
    maybe_trun_chunks = context_reorder(maybe_trun_chunks)
    section = "\n--New Chunk--\n".join([c for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response



@dataclass
class RAG:
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    working_dir: str = "rca_rag"
    storage_path: str = "adv_rag/53299511"
    embedding_batch_num: int = 32
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    tiktoken_model_name: str = "gpt-4o-mini"
    llm_model_max_async: int = 16
    llm_model_func: callable = llm_complete
    llm_model_kwargs: dict = field(default_factory=dict)
    embedding_function: EmbeddingFunc = field(default_factory=lambda: embedding)
    enable_llm_cache: bool = True
    embedding_func_max_async: int = 16
    llm_model_max_token_size: int = 8192
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
            )
        )
        
        self.embedding_function = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_function
        )
        
        self.full_docs = self.key_string_value_json_storage_cls(namespace="full_docs", global_config=asdict(self))
        self.text_chunks = self.key_string_value_json_storage_cls(namespace="text_chunks", global_config=asdict(self))
        self.chunks_vdb = self.vector_db_storage_cls(namespace="chunks",global_config=asdict(self),
                                                     embedding_func=self.embedding_function)
        self.chunk_entity_relation_graph = self.graph_storage_cls(namespace="chunk_entity_relation", global_config=asdict(self))
        self.entities_vdb = self.vector_db_storage_cls(namespace="entities",global_config=asdict(self),
                                                       embedding_func=self.embedding_function,
                                                       meta_fields={"entity_name"})
        self.relationships_vdb = self.vector_db_storage_cls(namespace="relationships",global_config=asdict(self),
                                                            embedding_func=self.embedding_function,
                                                            meta_fields={"src_id", "tgt_id"})
        
        
    async def query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
        
    
    async def rag_query(self, query: str, history: str, param: QueryParam = QueryParam()):
        if param.mode == "hybrid":
            response = await hybrid_query(
                    query,
                    history,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
        elif param.mode == "local":
            response = await local_query(
                    query,
                    history,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
        else:
            response = await naive_query(
                    query, 
                    history, 
                    self.chunks_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
        await self.query_done()
        return response


    def query_async(self, query: str, history: str, param: QueryParam = QueryParam()):
        return loop.run_until_complete(self.rag_query(query, history, param))





