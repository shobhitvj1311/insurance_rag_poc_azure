#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

if [ ! -d "$HOME/insurance-rag-venv" ]; then
    python3 -m venv "$HOME/insurance-rag-venv"
fi

source "$HOME/insurance-rag-venv/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p documents
mkdir -p rag_data
mkdir -p results

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
fi