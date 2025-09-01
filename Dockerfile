FROM python:3.11-slim
RUN pip install fastapi uvicorn hdbscan numpy
COPY app.py /app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
