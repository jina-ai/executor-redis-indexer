import operator

numeric_operators_redis = {
    'gte': operator.ge,
    'gt': operator.gt,
    'lte': operator.le,
    'lt': operator.lt,
    'eq': operator.eq,
    'ne': operator.ne,
}

def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches