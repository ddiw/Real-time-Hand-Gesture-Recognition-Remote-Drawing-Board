FROM python:3.12-slim

WORKDIR /app
# Dependencies first: editing app.py must not reinstall the wheel set.
COPY containers/0-web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY containers/0-web/app.py ./app.py
COPY containers/0-web/web ./web
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
