# syntax=docker/dockerfile:1.2
FROM python:3.8-slim

WORKDIR /workspace

COPY services/flower/requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "server.py"]