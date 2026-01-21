#!/bin/bash
# Lancer FastAPI et Streamlit dans le même conteneur

# Démarrer FastAPI en arrière-plan
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Démarrer Streamlit au premier plan
streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
