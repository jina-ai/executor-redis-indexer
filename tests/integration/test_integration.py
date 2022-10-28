import numpy as np
import pytest
from docarray import Document, DocumentArray
from executor import RedisIndexer
from helper import assert_document_arrays_equal, numeric_operators_redis
from jina import Flow


def test_flow(docker_compose):
    f = Flow().add(
        uses=RedisIndexer,
        uses_with={'index_name': 'test1', 'n_dim': 2},
    )

    with f:
        f.post(
            on='/index',
            inputs=[
                Document(id='a', embedding=np.array([1, 3])),
                Document(id='b', embedding=np.array([1, 1])),
                Document(id='c', embedding=np.array([3, 1])),
                Document(id='d', embedding=np.array([2, 3])),
            ],
        )

        docs = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1, 1]))],
        )
        assert docs[0].matches[0].id == 'b'


def test_reload_keep_state():
    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(
        uses=RedisIndexer,
        uses_with={'index_name': 'test2', 'n_dim': 3},
    )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)

    with f:
        second_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == len(second_search[0].matches)


def test_persistence(docs, docker_compose):
    f = Flow().add(
        uses=RedisIndexer,
        uses_with={'index_name': 'test3', 'n_dim': 2},
    )
    with f:
        f.index(docs)

    indexer = RedisIndexer(index_name='test3', distance='L2')
    assert_document_arrays_equal(indexer._index, docs)


@pytest.mark.parametrize(
    'filter_gen,operator',
    [
        (lambda threshold: f'@price:[{threshold} inf] ', 'gte'),
        (lambda threshold: f'@price:[({threshold} inf] ', 'gt'),
        (lambda threshold: f'@price:[-inf {threshold}] ', 'lte'),
        (lambda threshold: f'@price:[-inf ({threshold}] ', 'lt'),
        (lambda threshold: f'@price:[{threshold} {threshold}] ', 'eq'),
        (lambda threshold: f'(- @price:[{threshold} {threshold}]) ', 'ne'),
    ],
)
def test_filtering(filter_gen, operator, docker_compose):
    n_dim = 3

    f = Flow().add(
        uses=RedisIndexer,
        uses_with={
            'n_dim': n_dim,
            'columns': {'price': 'float'},
        },
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    with f:
        f.index(docs)

        for threshold in [10, 20, 30]:

            filter_ = filter_gen(threshold)
            doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
            indexed_docs = f.search(doc_query, parameters={'filter': filter_})

            assert len(indexed_docs[0].matches) > 0

            assert all(
                [
                    numeric_operators_redis[operator](r.tags['price'], threshold)
                    for r in indexed_docs[0].matches
                ]
            )
