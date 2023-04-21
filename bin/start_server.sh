#!/usr/bin/env bash
#
# Start up the flower server for training a model using federated learning

source bin/setup_environment.sh

docker-compose up flower
