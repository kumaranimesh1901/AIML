FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md alembic.ini ./
COPY configs ./configs
COPY src ./src
COPY dashboard ./dashboard

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[postgres,dashboard]" \
    && pip install --no-cache-dir scikit-learn umap-learn hdbscan scipy statsmodels

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
