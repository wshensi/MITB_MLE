FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jdk \
        gcc \
        g++ \
        procps \
        curl \
        bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install requirements FIRST
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import airflow; print(f'✓ Airflow {airflow.__version__} installed')" && \
    python -c "import sklearn; print(f'✓ scikit-learn {sklearn.__version__}')" && \
    python -c "import imblearn; print(f'✓ imbalanced-learn {imblearn.__version__}')"

# Create directory structure
RUN mkdir -p \
    /app/dags \
    /app/logs \
    /app/plugins \
    /app/utils \
    /app/models/model_artifacts \
    /app/models/model_metadata \
    /app/datamart/bronze \
    /app/datamart/silver \
    /app/datamart/gold/feature_label_store \
    /app/datamart/gold/predictions \
    /app/datamart/gold/monitoring_metrics \
    /app/data

ENV AIRFLOW_HOME=/app

# Copy utils folder (critical for ML pipeline)
COPY utils/ /app/utils/

# Copy dags folder
COPY dags/ /app/dags/

# Copy main Python files
COPY *.py /app/

# Copy data folder (comment out if you don't have a data folder)
# COPY data/ /app/data/

# Verify critical files are present
RUN echo "=== Verifying file structure ===" && \
    ls -la /app/ && \
    echo "" && \
    echo "=== Utils folder ===" && \
    ls -la /app/utils/ && \
    test -f /app/utils/__init__.py && echo "✓ __init__.py" || echo "✗ __init__.py MISSING" && \
    test -f /app/utils/model_training.py && echo "✓ model_training.py" || echo "✗ model_training.py MISSING" && \
    test -f /app/utils/model_inference.py && echo "✓ model_inference.py" || echo "✗ model_inference.py MISSING" && \
    test -f /app/utils/model_monitoring.py && echo "✓ model_monitoring.py" || echo "✗ model_monitoring.py MISSING" && \
    echo "" && \
    echo "=== Dags folder ===" && \
    ls -la /app/dags/ && \
    echo "" && \
    echo "=== Testing Python imports ===" && \
    python -c "import sys; sys.path.insert(0, '/app'); from utils.model_training import train_and_save_best_model; print('✓ Utils imports successful')" || echo "✗ Utils import FAILED"

EXPOSE 8080 8888

CMD ["bash"]
