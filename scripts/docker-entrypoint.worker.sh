#!/bin/sh
set -eu

HF_CACHE_DIR="${HF_HOME:-/models/huggingface}"
HF_CACHE_TEMPLATE_DIR="${HF_CACHE_TEMPLATE_DIR:-/opt/hf-cache-template}"

if [ -d "$HF_CACHE_TEMPLATE_DIR" ]; then
  mkdir -p "$HF_CACHE_DIR"
  if [ -z "$(find "$HF_CACHE_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
    echo "[entrypoint] seeding Hugging Face cache from $HF_CACHE_TEMPLATE_DIR"
    cp -a "$HF_CACHE_TEMPLATE_DIR"/. "$HF_CACHE_DIR"/
  fi
fi

exec "$@"
