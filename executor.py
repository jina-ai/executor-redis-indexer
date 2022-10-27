from jina import Executor, DocumentArray, requests


class RedisIndexer(Executor):
    """"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass