FROM python:3.12-slim

WORKDIR /app
COPY containers/3-pattern-command/requirements.txt ./requirements.txt
COPY containers/3-pattern-command ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8761
CMD ["python", "app.py"]
