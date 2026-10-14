import time

import faiss
import numpy as np


def l2_normalise(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def shuffle_order(n, seed):
    return np.random.default_rng(seed).permutation(n)


def build_flat_ip(gallery):
    gallery = np.ascontiguousarray(gallery, dtype=np.float32)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    return index


def build_hnsw(gallery, M, ef_construction, ef_search):
    gallery = np.ascontiguousarray(gallery, dtype=np.float32)
    index = faiss.IndexHNSWFlat(gallery.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(gallery)
    index.hnsw.efSearch = ef_search
    return index


def build_binary(codes, bits):
    index = faiss.IndexBinaryFlat(bits)
    index.add(np.ascontiguousarray(codes, dtype=np.uint8))
    return index


def timed_search(index, queries, k, repeats):
    queries = np.ascontiguousarray(queries)
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        best = min(best, time.perf_counter() - start)
    faiss.omp_set_num_threads(threads)
    return ids, 1000 * best / len(queries)


def timed_single_queries(index, queries, k, repeats, n_queries=200):
    queries = np.ascontiguousarray(queries[:n_queries])
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(len(queries)):
            index.search(queries[i:i + 1], k)
        best = min(best, time.perf_counter() - start)
    faiss.omp_set_num_threads(threads)
    return 1000 * best / len(queries)


def index_bytes(index):
    if isinstance(index, faiss.IndexBinary):
        return int(faiss.serialize_index_binary(index).nbytes)
    return int(faiss.serialize_index(index).nbytes)
