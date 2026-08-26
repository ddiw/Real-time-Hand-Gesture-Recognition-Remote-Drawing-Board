FROM python:3.12-slim

WORKDIR /app
# Dependencies first so a rule change rebuilds in seconds, not minutes.
COPY containers/3-pattern-command/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY containers/3-pattern-command ./

EXPOSE 8761
CMD ["python", "app.py"]
