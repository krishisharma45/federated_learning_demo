#!/usr/bin/env bash
#
# Lint source code

source bin/setup_environment.sh

docker-compose run --rm --entrypoint pylint "$SERVICE" --rcfile .pylintrc src/**/*.py tests/**/*.py