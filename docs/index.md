# pyHORDCOIN

Python interface to the HORDCOIN Julia package for Connected Information.

# Installing

Installation is easy, just:
1. (optional but very advised) create and activate a new virtual environment:
    ```bash
    pyhton -m venv hordcoin
    source hordcoin/bin/activate
    ```
2. change directory to the unpacked repository:
    ```bash
    cd /path/to/hordcoin
    ```
1. install the package:
    1. if you only want to use its components for your measures:
        ```bash
        pip install .
    1. if you want to contribute to the development: 
        ```bash
        pip install .[dev]
        ```
# Running
The first time you import the package it will take some time to install the Julia dependencies (and possibly Julia itself). From the second import on it's much faster, although the import of Julia still takes some time. This will be repaid during the maximisation of the entropies.

# Limitations
Not every functionality of the original library is available yet. The most notable is the possibility to call the Connected Information with precalculated entropies.
