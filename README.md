# DEPOCO

This repository implements the algorithms described in our paper [Deep Compression for Dense Point Cloud Maps](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf).

## How to get started (using Docker)

### Dependenices nvida-docker

Install nvida-docker and follow [these](https://stackoverflow.com/a/61737404)
instructions

## Data
You can download the dataset from [here](https://www.ipb.uni-bonn.de/html/projects/depoco/submaps.zip) and link the dataset to the docker container by configuring the Makefile

```sh
DATASETS=<path-to-your-data>
```

## Building the docker container

For building the Docker Container simply run 

```sh
make build
```

in the root directory.

## Running the Code

The first step is to run the docker container:

```sh
make run
```

The following commands assume to be run inside the docker container.

### Training

For training a network we first have to create the config file with all the parameters.
An example of this can be found in `/depoco/config/depoco.yaml`. 
Make sure to give each config file a unique `experiment_id: ...` to not override previous models.
To train the network simply run

```sh
python3 trainer -cfg <path-to-your-config>
```

### Evaluation

Evaluating the network on the test set can be done by:

```sh
python3 evaluate.py -cfg <path-to-your-config>
```

All results will be saved in a dictonary.

### Plotting the results

We can plot the quantitative results e.g. by using Jupyter-Lab.
An example of this is provided in `depoco/notebooks/visualize.ipynb`.
Jupyter-Lab can be started in the Docker container by:

```sh
jupyter-lab  --ip 0.0.0.0 --no-browser --allow-root
```

The 8888 port is forwarded which allows us to use it as if it would be on the host machine.

### Pretrained models

The config files and the pretrained weights of our models are stored in `depoco/network_files/eX/`. The results can be inspected by the jupyter notebook `depoco/notebooks/visualize.ipynb`.

## How to get started (without Docker)

### Installation

A list of all dependencies and install instructions can be derived from the Dockerfile.

### Running the code

After installation the training and evaluation can be run as explained before.

### Qualitative Results

Plotting the point clouds using open3d can be done by

```sh
pyhon3 evaluate -cfg <path-to-your-config>
```

This can **not** be done in the docker container and thus requires the installation on the local machine.


## Citation

If you use this library for any academic work, please cite the original paper.

```bibtex
@article{wiesmann2021ral,
author = {L. Wiesmann and A. Milioto and X. Chen and C. Stachniss and J. Behley},
title = {{Deep Compression for Dense Point Cloud Maps}},
journal = {IEEE Robotics and Automation Letters (RA-L)},
volume = 6,
issue = 2,
pages = {2060-2067},
doi = {10.1109/LRA.2021.3059633},
year = 2021
}
```
