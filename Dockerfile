FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FORTE_HOST=0.0.0.0 FORTE_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[vision]'
COPY config ./config
COPY scripts ./scripts
EXPOSE 8000
CMD ["forte-guard", "serve", "--host", "0.0.0.0"]
