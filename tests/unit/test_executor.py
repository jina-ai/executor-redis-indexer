import os

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.redis import DocumentArrayRedis
from executor import RedisIndexer
from helper import assert_document_arrays_equal

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '../docker-compose.yml'))


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(3)),
            Document(id='doc2', embedding=np.random.rand(3)),
            Document(id='doc3', embedding=np.random.rand(3)),
            Document(id='doc4', embedding=np.random.rand(3)),
            Document(id='doc5', embedding=np.random.rand(3)),
            Document(id='doc6', embedding=np.random.rand(3)),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', text='modified', embedding=np.random.rand(128)),
        ]
    )


def test_init(docker_compose):
    indexer = RedisIndexer(index_name='test')

    assert isinstance(indexer._index, DocumentArrayRedis)
    assert indexer._index._config.index_name == 'test'
    assert indexer._index._config.port == 6379


def test_index(docs, docker_compose):
    indexer = RedisIndexer(index_name='test1')
    indexer.index(docs)

    assert len(indexer._index) == len(docs)


def test_delete(docs, docker_compose):
    indexer = RedisIndexer(index_name='test2')
    indexer.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    indexer.delete({'ids': ids})
    assert len(indexer._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in indexer._index


def test_update(docs, update_docs, docker_compose):
    # index docs first
    indexer = RedisIndexer(index_name='test3')
    indexer.index(docs)
    assert_document_arrays_equal(indexer._index, docs)

    # update first doc
    indexer.update(update_docs)
    assert indexer._index['doc1'].text == 'modified'


def test_fill_embeddings(docker_compose):
    indexer = RedisIndexer(index_name='test4', distance='L2', n_dim=1)

    indexer.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    indexer.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1])).all()

    with pytest.raises(KeyError, match='b'):
        indexer.fill_embedding(DocumentArray([Document(id='b')]))


def test_filter(docker_compose):
    docs = DocumentArray.empty(5)
    docs[0].text = 'hello'
    docs[1].text = 'world'
    docs[2].tags['x'] = 0.3
    docs[2].tags['y'] = 0.6
    docs[3].tags['x'] = 0.8

    indexer = RedisIndexer(index_name='test5', columns={'text': 'str', 'x': 'float'})
    indexer.index(docs)

    result = indexer.filter(parameters={'filter': 'hello'})
    assert len(result) == 1
    assert result[0].text == 'hello'

    result = indexer.filter(parameters={'filter': '@x:[(0.5 inf] '})
    assert len(result) == 1
    assert result[0].tags['x'] == 0.8


@pytest.mark.parametrize(
    'metric, metric_name',
    [('L2', 'euclid_distance'), ('COSINE', 'cosine_distance')],
)
def test_search(metric, metric_name, docs, docker_compose):
    # test general/normal case
    indexer = RedisIndexer(index_name='test6', distance=metric)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [t[metric_name].value for t in doc.matches[:, 'scores']]
        print(f"similarities = {similarities}")
        assert sorted(similarities) == similarities
        assert len(similarities) == len(docs)


@pytest.mark.parametrize('limit', [1, 2, 3])
def test_search_with_match_args(docs, limit, docker_compose):
    indexer = RedisIndexer(index_name='test7', match_args={'limit': limit})
    indexer.index(docs)
    assert 'limit' in indexer._match_args.keys()
    assert indexer._match_args['limit'] == limit

    query = DocumentArray([Document(embedding=np.random.rand(128))])
    indexer.search(query)

    assert len(query[0].matches) == limit

    docs[0].tags['text'] = 'hello'
    docs[1].tags['text'] = 'world'
    docs[2].tags['text'] = 'hello'

    indexer = RedisIndexer(
        index_name='test8',
        columns={'text': 'str'},
        match_args={
            'filter': '@text:hello',
            'limit': 1,
        },
    )
    indexer.index(docs)

    indexer.search(query)
    assert len(query[0].matches) == 1
    assert query[0].matches[0].tags['text'] == 'hello'


def test_clear(docs, docker_compose):
    indexer = RedisIndexer(index_name='test9')
    indexer.index(docs)
    assert len(indexer._index) == 6
    indexer.clear()
    assert len(indexer._index) == 0


def test_columns(docker_compose):
    n_dim = 3
    indexer = RedisIndexer(index_name='test10', n_dim=n_dim, columns={'price': 'float'})

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
    indexer.index(docs)
    assert len(indexer._index) == 10
