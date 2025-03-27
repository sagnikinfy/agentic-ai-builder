import asyncio
import asyncpg
import nest_asyncio
import sqlalchemy
from sqlalchemy.future import select
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_connection
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession, async_scoped_session
from sqlalchemy.util import await_only
from pgvector.sqlalchemy import Vector
from pgvector.asyncpg import register_vector
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.base import LLM
#from sentence_transformers import SentenceTransformer
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Mapping
)
import enum
import uuid
nest_asyncio.apply()


class DistanceStrategy(str, enum.Enum):
    EUCLIDEAN = "l2",
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"

DEFAULT_DISTANCE = DistanceStrategy.COSINE
COLLECTION_NAME = "collection_sim"
Base = declarative_base()



### create schema ###

class BaseModelSch(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid = True), primary_key = True, default = uuid.uuid4)

class CollectionStore(BaseModelSch):
    __tablename__ = "vec_collection"
    name = sqlalchemy.Column(sqlalchemy.String)
    cmetadata = sqlalchemy.Column(JSON)
    embeddings = relationship(
        "EmbeddingStore",
        back_populates = "collection",
        passive_deletes = True
    )

    @classmethod
    async def get_by_name(cls, session: AsyncSession, name: str) -> Optional["CollectionStore"]:
        q = select(cls).filter(cls.name == name)
        res = await session.execute(q)
        return res.scalar()

    @classmethod
    async def get_or_create(cls, session: AsyncSession, name: str,
                            cmetadata: Optional[dict] = None) -> Tuple["CollectionStore", bool]:
        created = False
        collection = await cls.get_by_name(session, name)

        if collection:
            return (collection, created)

        collection = cls(name = name, cmetadata = cmetadata)
        session.add(collection)
        await session.commit()
        created = True
        return (collection, created)

class EmbeddingStore(BaseModelSch):
    __tablename__ = "embd_table"
    collection_id = sqlalchemy.Column(
        UUID(as_uuid = True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete = "CASCADE",
        ),
    )
    collection = relationship(CollectionStore, back_populates = "embeddings")
    embeddings: Vector = sqlalchemy.Column(Vector(None))
    document = sqlalchemy.Column(sqlalchemy.String, nullable = True)
    cmetadata = sqlalchemy.Column(JSON, nullable = True)
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable = True)


async def anext(ait):
    return await ait.__anext__()



### dbapi ###

class AsyncPGVectorDriver(VectorStore):
    def __init__(self, pg_conn: str, embd_model: Embeddings, collection_name: str = COLLECTION_NAME,
                      collection_metadata: Optional[dict] = None, distance_strat:
                       DistanceStrategy = DEFAULT_DISTANCE, pre_del_collection: bool = False,
                      relevane_score_fn: Optional[Callable[[float], float]] = None) -> None:
        super().__init__()
        self.pg_conn = pg_conn
        self.embd_model = embd_model
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strat = distance_strat
        self.pre_del_collection = pre_del_collection
        self.relevane_score_fn = relevane_score_fn
        self.__post__init()


    def __post__init(self) -> None:
        self._engine = self.create_engine()
        self._async_session = async_scoped_session(sessionmaker(self._engine, expire_on_commit = True, class_ = AsyncSession),
                                                   scopefunc = asyncio.current_task)
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore


    def create_engine(self) -> AsyncEngine:
        return create_async_engine(
            self.pg_conn,
            echo = True,
            future = True
        )

    async def get_db_session(self) -> Any:
        session = self._async_session()
        try:
            yield session
        except Exception as e:
            print("Session rollback because of exception")
            session.rollback()
            raise
        #finally:
            #await session.close()


    async def create_vec_ext(self) -> None:
        ag = self.get_db_session()
        session = await anext(ag)
        await session.execute(text("create extension if not exists vector"))
        await session.commit()


    async def create_table(self) -> None:
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            print(f"Error occurred while creating table => {e}")


    async def drop_table(self) -> None:
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception as e:
            print(f"Error occurred while deleting table => {e}")



    async def create_coll(self) -> None:
        try:
            if(self.pre_del_collection):
                await delete_coll()
            ag = self.get_db_session()
            session = await anext(ag)
            await CollectionStore.get_or_create(
                session, self.collection_name, cmetadata = None
            )
        except Exception as e:
            print(f"Error occurred while creating collection => {e}")


    async def delete_coll(self) -> None:
        try:
            ag = self.get_db_session()
            session = await anext(ag)
            collection = await self.get_collection(session)
            if(not collection):
                print("collection not found")
                return
            await session.delete(collection)
            await session.commit()
        except Exception as e:
            print(f"Error occurred while deleting collection => {e}")


    async def get_collection(self, session: AsyncSession) -> Optional["CollectionStore"]:
        return await self.CollectionStore.get_by_name(session, self.collection_name)


    async def add_embeddings(self, texts: Iterable[str], embeddings: List[List[float]],
                             metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None,
                             **kwargs: Any) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        ag = self.get_db_session()
        session = await anext(ag)
        collection = await self.get_collection(session)
        if(not collection):
            raise ValueError("Collection not found")
        for txt, mtd, emb, i in zip(texts, metadatas, embeddings, ids):
            embd_store = EmbeddingStore(
                embeddings = emb,
                document = txt,
                cmetadata = mtd,
                custom_id = i,
                collection_id = collection.uuid
            )
            session.add(embd_store)
        await session.commit()

        return ids


    async def insert_docs(self, documents: List[Document],
                          collection_name: str = COLLECTION_NAME,
                          distance_strategy: DistanceStrategy = DEFAULT_DISTANCE,
                          ids: Optional[List[str]] = None, pre_del_col: bool = False, **kwargs: Any) -> None:

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        embeddings = self.embd_model.embed_documents(list(texts))
        await self.add_embeddings(texts = texts, embeddings = embeddings, metadatas = metadatas)

    

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strat == "l2":
            return self.EmbeddingStore.embeddings.l2_distance
        elif self._distance_strat == "cosine":
            return self.EmbeddingStore.embeddings.cosine_distance
        elif self._distance_strat == "inner":
            return self.EmbeddingStore.embeddings.max_inner_product
        else:
            raise ValueError(
                f"Unexpected distance value => {self._distance_strat}"
            )



    async def similarity_search_with_score_main(self, embedding: List[float], k: int = 4,
                                           filter: Optional[dict] = None) -> List[Tuple[Document, float]]:

        ag = self.get_db_session()
        session = await anext(ag)
        collection = await self.get_collection(session)
        if not collection:
            raise ValueError("Collection not found")

        filter_by = self.EmbeddingStore.collection_id == collection.uuid

        if(filter is not None):
            filter_clauses = []
            for key, value in filter.items():
                IN = "in"
                if(isinstance(value, dict) and IN in map(str.lower, value)):
                    value_case_insensitive = {
                        k.lower(): v for k,v in value.items()
                    }
                    filter_by_metadata = EmbeddingStore.cmetadata[
                        key
                    ].astext.in_(value_case_insensitive[IN])
                    filter_clauses.append(filter_by_metadata)
                else:
                    filter_by_metadata = EmbeddingStore.cmetadata[
                        key
                    ].astext == str(value)
                    filter_clauses.append(filter_by_metadata)

            filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

        _type = self.EmbeddingStore

        q = select(self.EmbeddingStore, self.distance_strategy(embedding).label(
            "distance"
        )).filter(filter_by).order_by(sqlalchemy.asc("distance")).join(
            self.CollectionStore,
            self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
        ).limit(k)

        res = await session.execute(q)
        results: List[Any] = res.all()


        docs = [
            (
                Document(
                    page_content = result.EmbeddingStore.document,
                    metadata = result.EmbeddingStore.cmetadata,
                ),
                result.distance if self.embd_model is not None else None,
            )
            for result in results
        ]


        return docs


    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any) -> List[Document]:

        embedding = self.embd_model.embed_query(query)
        loop = asyncio.get_event_loop()
        nest_asyncio.apply(loop)
        docs_with_score = loop.run_until_complete(self.similarity_search_with_score_main(embedding = embedding, k = k, filter = filter))

        return [doc for doc, _ in docs_with_score]


    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:

        embedding = self.embd_model.embed_query(query)
        loop = asyncio.get_event_loop()
        nest_asyncio.apply(loop)
        docs = loop.run_until_complete(self.similarity_search_with_score_main(embedding = embedding, k = k, filter = filter))
        return docs


    @classmethod
    async def from_texts(cls: Type[PGVector], texts: List[str], embedding, Embeddings,
                                        metadatas: Optional[List[dict]] = None,
                                        collection_name: str = COLLECTION_NAME,
                                        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE,
                                        ids: Optional[List[str]] = None,
                                        pre_del_col: bool = False, **kwargs: Any) -> PGVector:
        return



    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:

        embeddings = self.embd_model.embed_documents(list(texts))
        return self.add_embeddings(
            texts = texts, embeddings = embeddings, metadatas = metadatas, ids = ids, **kwargs
        )


    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.relevane_score_fn is not None:
            return self.relevane_score_fn
        
        if self._distance_strat == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strat == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strat == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                f"No supported normalized function for {self.distance_strategy}" 
            )
