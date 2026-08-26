FROM python:3.12-slim

WORKDIR /app
COPY containers/0-web/requirements.txt ./requirements.txt
COPY containers/0-web/app.py ./app.py
COPY containers/0-web/web ./web
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
