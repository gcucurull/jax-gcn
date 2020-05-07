# Graph Convolutional Networks in JAX

This repository implements GCNs in JAX (check it out on [github](https://github.com/google/jax)). The code contains the model definition of a Graph Convolutional Network with two graph convolutional layers, following the model used in the paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

## Usage
Run 

```python train.py```

to train a model on the Cora dataset.

## Good to know
I implemented a sparse matrix multiplication function to support sparse adjacency matrices, which is enabled by default. If you get any error with it, it can be disabled by adding the flag `--no-sparse` to the run command.

## Cite
This is an implementation in JAX of the [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) paper. If you use it in your research, please cite the paper:
```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
