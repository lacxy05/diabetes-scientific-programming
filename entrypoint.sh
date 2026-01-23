#!/bin/bash
# Launch FastAPI and Streamlit in the same container

# Start FastAPI in the background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run src/06_dashboard.py --server.port 8501 --server.address 0.0.0.0
