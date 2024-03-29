# highdimzo
Zeroth-Order Optimization of Composite Objectives
# library
A library for exponeitated update based stochastic optimisation algorithms and experiments on them.

## Dependencies
In order to run the experiments the following software is necessary:

| Software     | Importance           | Installation Instruction                                              |
|--------------|----------------------|-----------------------------------------------------------------------|
| Python       | Necessary            | https://wiki.python.org/moin/BeginnersGuide/Download                  |
| MXNet        | Necessary            | https://mxnet.apache.org/versions/1.7.0/get_started?                  |
| Numpy        | Necessary            | https://numpy.org/install/                                            |
| SciPy        | Necessary            | https://scipy.org/ |

- Example:   
  - python experiment_cem_zo.py -d -1 --mode PN --data MNIST --l1 -4 --l2 -4 --alg PSGD
  - python experiment_cem_zo.py -d -1 --mode PP --data MNIST --l1 -4 --l2 -4 --alg ExpGrad
  - python experiment_cem_zo.py -d -1 --mode PN --data CIFAR --l1 -1 --l2 -1 --alg AdaExpGrad
  - python experiment_cem_zo.py -d -1 --mode PP --data CIFAR --l1 -1 --l2 -1 --alg AdaExpGradP
- Parameters
    - d -1: CPU, 0: GPU 1, 1 : GPU 2
    - mode PP or PN
    - data dataset MNIST or CIFAR
    - l1 l1 regularization
    - l2 l2 regularization
    - kappa controlling the lower bound of the loss
    - alg choice of algorithm: PSGD|ExpGrad|AdaExpGrad|AdaExpGradP|ExpStorm|AccZOM|AOExpGrad
- Experiment results can be ploted using the script plot_csv_zo_cifar.ipynb or plot_csv_zo_mnist.ipynb
