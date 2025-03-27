import os
import json
import base64
import hashlib
from uuid import uuid4
import numpy as np
from typing import TypedDict, Literal, Union, Callable
from dataclasses import dataclass, asdict
import sqlite3
from google.cloud import storage
from google.oauth2 import service_account
import logging
from logging import getLogger

logging.basicConfig(level=logging.INFO)

f_ID = "__id__"
f_VECTOR = "__vector__"
f_METRICS = "__metrics__"
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)
Float = np.float32
ConditionLambda = Callable[[Data], bool]
logger = getLogger("nano-vectordb")

creds_bq = service_account.Credentials.from_service_account_file(
                "apigee.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])
client = storage.Client(credentials = creds_bq, project = "apigee-infosys")


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def read_from_storage(bucket: str, file: str) -> Union[str, None]:
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(file)
    if blob is None:
        return None
    else:
        return blob.download_as_string()


def load_storage(bucket: str, file: str) -> Union[DataBase, None]:
    data = read_from_storage(bucket, file)
    if data is None:
        return None
    data = json.loads(data)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    bucket: str = "sim_cases"
    file: str = "db.json"

    def pre_process(self):
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.bucket, self.file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        self.__storage["additional_data"] = kwargs

    def get(self, ids: list[str]):
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def __len__(self):
        return len(self.__storage["data"])

    def query(
        self,
        query: np.ndarray,
        top_k: int = 10,
        better_than_threshold: float = None,
        filter_lambda: ConditionLambda = None,
    ) -> list[dict]:
        return self.usable_metrics[self.metric](
            query, top_k, better_than_threshold, filter_lambda=filter_lambda
        )

    def _cosine_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float,
        filter_lambda: ConditionLambda = None,
    ):
        query = normalize(query)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]
        scores = np.dot(use_matrix, query)
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]
        sort_abs_index = filter_index[sort_index]
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            if (
                better_than_threshold is not None
                and scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
        return results
