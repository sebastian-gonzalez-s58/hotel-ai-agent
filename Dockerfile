FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY hotel_guest_assistant_agent_openai.py .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --workers ${WEB_CONCURRENCY:-2} --bind 0.0.0.0:${PORT:-8000} --timeout ${WORKER_TIMEOUT_SECONDS:-60}"]
