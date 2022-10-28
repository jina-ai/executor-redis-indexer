import numpy as np
from docarray import Document
from executor import RedisIndexer
from jina import Flow


def test_replicas(docker_compose):
    n_dim = 10

    f = Flow().add(
        uses=RedisIndexer,
        uses_with={
            'index_name': 'test1',
            'n_dim': n_dim,
            'distance': 'L2',
            'ef_construction': 256,
            'm': 32,
            'ef_runtime': 256,
        },
    )

    docs_index = [
        Document(id=str(i), embedding=np.random.random(n_dim)) for i in range(1000)
    ]

    docs_query = docs_index[:100]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=RedisIndexer,
        uses_with={'index_name': 'test1', 'n_dim': n_dim, 'update_schema': False},
        replicas=4,
    )

    with f_with_replicas:
        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas


def test_replicas_reindex(docker_compose):
    n_dim = 10

    f = Flow().add(
        uses=RedisIndexer,
        uses_with={
            'index_name': 'test2',
            'n_dim': n_dim,
            'distance': 'L2',
            'ef_construction': 256,
            'm': 32,
            'ef_runtime': 256,
        },
    )

    docs_index = [
        Document(id=f'd{i}', embedding=np.random.random(n_dim)) for i in range(1000)
    ]

    docs_query = docs_index[:100]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=RedisIndexer,
        uses_with={'index_name': 'test2', 'n_dim': n_dim, 'update_schema': False},
        replicas=4,
    )

    with f_with_replicas:
        f_with_replicas.post(on='/index', inputs=docs_index, request_size=1)

        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas
