#/bin/bash

uv run train.py
uvicorn app.api:app --host "0.0.0.0"
