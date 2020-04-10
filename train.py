import argparse
import time

import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers

from utils import load_data
from models import GCN


def loss(params, batch):
    inputs, targets, adj, is_training, rng = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    return -np.mean(np.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets, adj, is_training, rng = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict_fun(params, inputs, adj, is_training=is_training, rng=rng), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    rng_key = random.PRNGKey(0)
    dropout = 0.5
    step_size = 0.01
    hidden = 16
    num_epochs = 200
    n_nodes = adj.shape[0]
    n_feats = features.shape[1]


    init_fun, predict_fun = GCN(nhid=hidden, 
                                nclass=labels.shape[1],
                                dropout=dropout)
    input_shape = (-1, n_nodes, n_feats)
    _, init_params = init_fun(rng_key, input_shape)

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    #@jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    opt_state = opt_init(init_params)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        batch = (features, labels, adj, True, rng_key)
        opt_state = update(epoch, opt_state, batch)
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        eval_batch = (features, labels, adj, False, rng_key)
        train_loss = loss(params, eval_batch)
        train_acc = accuracy(params, eval_batch)
        test_acc = accuracy(params, eval_batch)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set loss {}".format(train_loss))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))



    # preds = predict_fun(init_params, features, adj, rng=rng_key, is_training=True)
