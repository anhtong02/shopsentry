FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY simulator/requirements.txt simulator/requirements.txt
RUN pip install --no-cache-dir -r simulator/requirements.txt

COPY simulator/ simulator/

CMD ["python", "-m", "simulator.run", "--scenario", "mixed"]