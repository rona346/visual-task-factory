FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["sh", "-c", "python -m visual_task_factory.demo_app --host 0.0.0.0 --port ${PORT:-8080}"]
