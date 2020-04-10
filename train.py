import argparse

from jax import jit, grad, random

from utils import load_data
from models import GCN


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    rng_key = random.PRNGKey(0)
    dropout = False
    hidden = 16
    n_nodes = adj.shape[0]
    n_feats = features.shape[1]


    init_fun, predict_fun = GCN(nhid=hidden, 
                                nclass=labels.max()+1,
                                dropout=dropout)
    input_shape = (-1, n_nodes, n_feats)
    _, init_params = init_fun(rng_key, input_shape)


    preds = predict_fun(init_params, features, adj)


def train():
    pass

if __name__ == "__main__":
    main()