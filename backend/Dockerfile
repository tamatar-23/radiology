FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY ai/ /app/ai/
COPY data/samples/ /app/data/samples/

RUN mkdir -p /app/data/uploads

ENV PYTHONPATH=/app
ENV AI_MODE=mock

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
