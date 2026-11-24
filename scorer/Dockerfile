FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY scorer/requirements.txt scorer/requirements.txt
RUN pip install --no-cache-dir -r scorer/requirements.txt

COPY scorer/ scorer/

CMD ["python", "-m", "scorer.scorer"]