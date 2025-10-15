#!/bin/bash
set -eu
SCRIPT_FILE="$(cd "$(dirname "$0")"; pwd)/$(basename "$0")"

cd $(dirname $SCRIPT_FILE)

rm -f .env
touch .env
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
