#!/usr/bin/env bash
# Usage: ./deploy.sh ubuntu@<elastic-ip>
# Syncs code + enabled model weights to the instance and (re)starts the stack.
# keys.yaml, .env and data/ live only on the server and are never overwritten.
set -euo pipefail
HOST=${1:?usage: ./deploy.sh user@host}
uv run pytest -q
rsync -az --delete --exclude .git --exclude .venv --exclude data --exclude runs --exclude labeled \
  --exclude keys.yaml --exclude .env --exclude '*-int8.onnx' --exclude 'adamcodd*' --exclude 'marqo*' \
  --exclude 'nudenet-640m*' ./ "$HOST:ctc/"
ssh "$HOST" 'cd ctc && test -f .env && test -f keys.yaml && docker compose up -d --build && sleep 5 && docker compose ps'
