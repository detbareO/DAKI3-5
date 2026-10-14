import numpy as np


def fit_on_train(features, roles, max_rows, seed):
    train = features[np.asarray(roles) == "train"]
    mean = train.mean(axis=0, dtype=np.float64)

    if len(train) > max_rows:
        rng = np.random.default_rng(seed)
        train = train[rng.choice(len(train), size=max_rows, replace=False)]

    # PCA: egenvektorer af kovariansmatricen
    centred = train.astype(np.float64) - mean
    covariance = centred.T @ centred / (len(centred) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]  # størst varians først
    return mean.astype(np.float32), eigenvectors[:, order].astype(np.float32)


def lsh_matrix(dim, bits, seed):
    return np.random.default_rng(seed).standard_normal((dim, bits)).astype(np.float32)


def to_bits(features, mean, projection):
    # centrer, ellers er alle features >= 0 pga. ReLU
    return (features - mean) @ projection > 0


def encode(features, mean, projection):
    return np.packbits(to_bits(features, mean, projection), axis=1)
