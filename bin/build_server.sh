#!/usr/bin/env bash
#
# Build Docker image for flower server

source bin/setup_environment.sh

docker-compose build flower
