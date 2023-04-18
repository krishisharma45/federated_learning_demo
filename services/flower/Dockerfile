# syntax=docker/dockerfile:1.2
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

COPY services/flower/requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.server"]