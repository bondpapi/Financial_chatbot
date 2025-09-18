# Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but helpful for builds/certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tzdata && \
    rm -rf /var/lib/apt/lists/*

# Leverage layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY .env ./env
# Copy app code
COPY . ./

# Streamlit: bind to 0.0.0.0 and use $PORT for cloud, 8501 locally
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run
CMD ["bash","-lc","streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
