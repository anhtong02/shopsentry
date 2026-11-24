FROM python:3.13-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app 
ENV PYTHONUNBUFFERED=1

COPY pipeline/requirements.txt pipeline/requirements.txt
RUN pip install --no-cache-dir -r pipeline/requirements.txt

COPY pipeline/ pipeline/
COPY feature_repo/ feature_repo/

CMD ["python", "-m", "pipeline.spark_streaming"]