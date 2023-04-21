<h1>Federated Learning Demo</h1>

This repo contains demo code to show how the Python package Flower can easily stand up a federated learning training pipeline with three distinct clients using the MNIST dataset. The repo showcases federated evaluation techniques and applies the FedAvg aggregation formula to average the model weights for each round.

[Python 3.8][python-url]

### Requirements

- [Docker][docker-url]
- [Docker Compose][docker-compose-url]
- [NVIDIA Docker Container Runtime][nvidia-url]

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes.

Docker is used to ensure consistency across development, test, training, and production
environments. The Docker image is fully self-contained, so all required frameworks and libraries
will be installed as part of the image creation process.

### First Time Users

If this is your first time using this application, please ensure that you have installed the
[requirements](#requirements) listed above before proceeding. If you are using MacOS or Linux, you
can run this command:

```sh
brew install git docker
```

On Windows you can run this commmand:

```sh
choco install git docker
```

To get started, start playing with some of the [commands](#summary-of-commands) or [launch the
application locally](#launching-the-application).

### Launching the Application

To launch this application, you need to first build the Docker image using

```sh
bin/build_server.sh
```

and then bring up the Flower server for the application with

```sh
bin/start_server.sh
```

Once you're done working with your Flower server, you can stop all containers and remove the
containers, volumes and images associated with the application by running:

```sh
bin/stop_server.sh
```

Below are additional instruction on [training](#training) and [running tests](#testing) to verify
that everything is working.


### Summary of Commands

The `bin/` directory contains basic shell bin that allow us access to common commands on most
environments. We're not guaranteed much functionality on any generic machine, so keeping these
basic is important.

The most commonly used bin are:

- `bin/build_server.sh` - build docker image for the Flower server container
- `bin/start_server.sh` - starts docker container for the Flower server
- `bin/stop_server.sh` - stops docker container for the Flower server
- `bin/connect_client.sh` - connects a new instance of a Flower client to the Flower server
- `bin/setup_environment.sh` - sets up the environment variables for the Docker container
- `bin/train.sh` - train a model


[license-url]: ./LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/kungfuai/
[python-url]: https://www.python.org
[docker-url]: https://www.docker.com
[docker-compose-url]: https://docs.docker.com/compose/install/
[nvidia-url]: https://github.com/NVIDIA/nvidia-container-runtime
[kungfu-shield]: https://img.shields.io/badge/KUNGFU.AI-2022-red
[kungfu-url]: https://www.kungfu.ai
