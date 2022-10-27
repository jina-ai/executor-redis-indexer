from jina import Executor, DocumentArray, requests
from typing import Optional, Dict, Any, List, Tuple, Union
from jina.logging.logger import JinaLogger


class RedisIndexer(Executor):
    """RedisIndexer indexes Documents into a Redis server using DocumentArray  with `storage='redis'`"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        index_name: str = 'persisted',
        update_schema: bool = True,
        distance: str = 'COSINE',
        n_dim: int = 128,
        match_args: Optional[Dict] = None,
        redis_config: Optional[Dict[str, Any]] = None, #TODO is None or empty dict
        index_text: bool = False,
        tag_indices: Optional[List[str]] = None, #TODO is None or empty list
        batch_size: int = 64,
        method: str = 'HNSW',
        ef_construction: Optional[int] = None,
        m: Optional[int] = None,
        ef_runtime: Optional[int] = None,
        block_size: Optional[int] = None,
        initial_cap: Optional[int] = None,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        **kwargs,
    ):
        """
        :param host: Hostname of the Redis server
        :param port: Port of the Redis server
        :param index_name: Redis Index name used for the storage
        :param update_schema: If set to True, Redis will update search schema.
        :param distance: The distance metric used for the vector index and vector search
        :param n_dim: Number of dimensions
        :param match_args: The arguments to `DocumentArray`'s match function
        :param redis_config: Additional Redis client configuration object
        :param index_text: If set to True, Redis will index the text attribute of each Document to allow text search.
        :param tag_indices: Tag fields to be indexed in Redis to allow text search on them.
        :param batch_size: Batch size used to handle storage batch operations.
        :param method: Vector similarity index algorithm in Redis, either FLAT or HNSW.
        :param ef_construction: Number of maximum allowed potential outgoing edges candidates for each node in the graph, during the graph building. Defaults to the default `EF_CONSTRUCTION` value in Redis.
        :param m: Number of maximum allowed outgoing edges for each node in the graph in each layer. Defaults to the default `M` value in Redis.
        :param ef_runtime: Number of maximum top candidates to hold during the KNN search. Defaults to the default `EF_RUNTIME` value in Redis.
        :param block_size: Block size to hold BLOCK_SIZE amount of vectors in a contiguous array. Defaults to the default `BLOCK_SIZE` value in Redis.
        :param initial_cap: Initial vector capacity in the index affecting memory allocation size of the index. Defaults to the default `INITIAL_CAP` value in Redis.
        :param columns: precise columns for the Indexer (used for filtering).
        """
        super().__init__(**kwargs)
        self._match_args = match_args or {}

        self._index = DocumentArray(
            storage='redis',
            config={
                'n_dim': n_dim,
                'host': host,
                'port': port,
                'index_name': index_name,
                'update_schema': update_schema,
                'distance': distance,
                'redis_config': redis_config or {},
                'index_text': index_text,
                'tag_indices': tag_indices or [],
                'batch_size': batch_size,
                'method': method,
                'ef_construction': ef_construction,
                'm': m,
                'ef_runtime': ef_runtime,
                'block_size': block_size,
                'initial_cap': initial_cap,
                'columns': columns,
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        """Index new documents
        :param docs: the Documents to index
        """
        self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Dict = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint

        """
        
        match_args = (
                {**self._match_args, **parameters}
                if parameters is not None
                else self._match_args
            )
        docs.match(self._index, **match_args)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters of the request

        Keys accepted:
            - 'ids': List of Document IDs to be deleted
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/filter')
    def filter(self, parameters: str, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` using Redis: https://docarray.jina.ai/advanced/document-store/redis/#search-by-filter-query
        :param parameters: parameters of the request, containing the `filter` query
        """
        return self._index.find(filter=parameters.get('filter', None))

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Fill embedding of Documents by id

        :param docs: DocumentArray to be filled with Embeddings from the index
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index"""
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
