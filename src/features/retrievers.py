from qdrant_client import models


def sparse_queary(vector: dict, limit: int = 100) -> models.Prefetch:
    """Формирует запрос для поиска по разряженному вектору
    Args:
        vector (dict): запись (которую получаем от Qdrant)
        limit (int, optional): _description_. Сколько должен вернуть записей поиск.
        Defaults to 100

    Returns:
        models.Prefetch: класс для дальнейшего использования в query_points (QdrantClient)
    """
    return models.Prefetch(
        query=models.SparseVector(
            indices=vector["text-sparse"].indices,
            values=vector["text-sparse"].values,
        ),
        using="text-sparse",
        limit=limit,
    )


def dense_query(vector: dict, limit: int = 100) -> models.Prefetch:
    """Формирует запрос для поиска по плотному вектору
    Args:
        vector (dict): запись (которую получаем от Qdrant)
        limit (int, optional): _description_. Сколько должен вернуть записей поиск.
        Defaults to 100

    Returns:
        models.Prefetch: класс для дальнейшего использования в query_points (QdrantClient)
    """

    return models.Prefetch(query=vector["dense"], using="dense", limit=limit)


def dense_sparse_query(
    vector: dict, limit_dense: int = 1000, limit_sparse: int = 100
) -> models.Prefetch:
    """Формирует запрос для гибридного поиска. Сначала осуществляется поиск по плотному вектору,
    потом по полученным записям идет поиск по разряженному вектору

    Args:
        vector (dict): запись (которую получаем от Qdrant)
        limit_dense (int, optional): сколько записей вернет поиск по разряженным векторам.
        Defaults to 1000.
        limit_sparse (int, optional): сколько записей вернут поиск по плотному вектору.
        Defaults to 100.

    Returns:
        models.Prefetch: _description_
    """
    return models.Prefetch(
        prefetch=[dense_query(vector, limit_dense)],
        query=models.SparseVector(
            indices=vector["text-sparse"].indices,
            values=vector["text-sparse"].values,
        ),
        using="text-sparse",
        limit=limit_sparse,
    )
