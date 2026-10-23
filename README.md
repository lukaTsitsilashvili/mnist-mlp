# MNIST MLP in PyTorch

A simple multilayer perceptron (784 -> 128 -> 64 -> 10) that classifies
handwritten digits from MNIST. Built in pure PyTorch as a CS229 practice project.

## Run

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python mnist_mlp.py

## Result

About 97-98% test accuracy after 5 epochs on CPU.
