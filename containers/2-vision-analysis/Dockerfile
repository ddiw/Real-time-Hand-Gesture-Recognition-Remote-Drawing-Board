FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY containers/2-vision-analysis/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 모델 번들은 이미지 빌드 시점에 내려받아 포함한다.
# 컨테이너 기동 시 외부 다운로드에 의존하면 네트워크 장애가 서비스 장애로 이어진다 (PRD 2.4).
ARG HAND_LANDMARKER_MODEL_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
RUN mkdir -p models \
    && curl -fL "$HAND_LANDMARKER_MODEL_URL" -o models/hand_landmarker.task

COPY containers/2-vision-analysis/app ./app

ENV HAND_LANDMARKER_MODEL_PATH=/app/models/hand_landmarker.task \
    PYTHONUNBUFFERED=1

# 8760: Container A 프레임 수신(ingest), 8762: /health, /metrics
EXPOSE 8760 8762

CMD ["python", "-m", "app.main"]
