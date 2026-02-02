# pyHordcoin - Python interface for Hordcoin

This is a python interface for the Julia package Hordcoin which provides methods for finding probability distributions with maximal Shannon entropy given a fixed marginal distribution or entropy up to a chosen order, and to compute the Connected Information. 

# Installing

Installation is easy, just:
1. (optional but very advised) create and activate a new virtual environment:
    ```bash
    pyhton -m venv hordcoin
    source hordcoin/bin/activate
    ```
2. change directory to the unpacked repository:
    ```bash
    cd /path/to/pyhordcoin
    ```
1. install the package:
    1. if you only want to use its components for your measures:
        ```bash
        pip install .
        ```
    1. if you want to contribute to the development: 
        ```bash
        pip install .[dev]
        ```

# Running
The first time you import the package it will take some time to install the Julia dependencies (and possibly Julia itself). From the second import on it's much faster, although the import of Julia still takes some time. This will be repaid during the maximisation of the entropies.

# Reading the documentation
If I didn't upload it somewhere else, from the pyHordcoin folder run:
```bash
pip install -r docs/requirements.txt
```
to get the right packages.

Now you can read the documentation running:
```bash
mkdocs serve
```
And opening your browser at: http://127.0.0.1:8000/.
