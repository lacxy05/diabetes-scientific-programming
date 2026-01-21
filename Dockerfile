FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas scikit-learn fastapi uvicorn joblib streamlit requests

EXPOSE 8000
EXPOSE 8501

# Lancer le script d'entrée
CMD ["./entrypoint.sh"]
