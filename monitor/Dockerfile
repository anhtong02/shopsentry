FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY monitor/requirements.txt monitor/requirements.txt
RUN pip install --no-cache-dir -r monitor/requirements.txt

COPY monitor/ monitor/

EXPOSE 8001
CMD ["python", "-m", "monitor.monitor"]