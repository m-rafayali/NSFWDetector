#!/usr/bin/env bash
# Usage: ./deploy.sh ubuntu@<elastic-ip>
# Syncs code + enabled model weights to the instance and (re)starts the stack.
# .env (incl. CTC_JWT_SECRET) lives only on the server and is never overwritten.
set -euo pipefail
HOST=${1:?usage: ./deploy.sh user@host}
uv run pytest -q
rsync -az --delete --exclude .git --exclude .venv --exclude data --exclude runs --exclude labeled \
  --exclude .env --exclude models ./ "$HOST:ctc/"
# only the weights config.yaml enables (~0.2 GB)
uv run python -c "import yaml; [print(m.get('file', n) + '.onnx') for n, m in yaml.safe_load(open('config.yaml'))['models'].items() if m.get('enabled')]" > /tmp/ctc-models.txt
rsync -az --delete --files-from=/tmp/ctc-models.txt models/ "$HOST:ctc/models/"
ssh "$HOST" 'cd ctc && grep -q "^CTC_JWT_SECRET=." .env && podman-compose up -d --build --force-recreate && sleep 5 && podman ps'
