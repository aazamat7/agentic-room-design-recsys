FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY renovation_agent ./renovation_agent
COPY scripts ./scripts

EXPOSE 8000
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "."]
