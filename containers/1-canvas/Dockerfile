FROM python:3.12-slim

WORKDIR /app
# Dependencies first: editing canvas or pattern-command sources must not
# reinstall opencv and numpy on every build.
COPY containers/1-canvas/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY containers/1-canvas/app.py containers/1-canvas/drawing_canvas.py ./
COPY containers/3-pattern-command ./3-pattern-command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8770"]
