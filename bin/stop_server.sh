#!/usr/bin/env bash
#
# Stop and remove the container and image associated with the flower server

source bin/setup_environment.sh

docker-compose down flower
