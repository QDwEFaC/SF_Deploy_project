FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    FEATURE_COLUMNS_PATH=/app/configs/feature_columns.json
	
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*
	
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
	&& pip install --no-cache-dir uwsgi
	
COPY src/ /app/src/
COPY models/ /app/models/
COPY configs/ /app/configs/
COPY uwsgi.ini /app/uwsgi.ini

CMD ["uwsgi", "--ini", "/app/uwsgi.ini", "--http-socket", "0.0.0.0:3031"]