#!/usr/bin/env bash
#
# Connect a new instance of a flower client to the flower server

source bin/setup_environment.sh

docker-compose run --rm flower python -m src.client "$@"