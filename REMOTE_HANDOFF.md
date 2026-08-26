# 휴대폰/ngrok 원격 그림판 인수인계

## 구현 상태

`Real-time-Cookie-Defect-Detection-System`의 모바일 촬영/대시보드 구조를 참고해
이 프로젝트용 원격 실행 경로를 추가했다. 기존 로컬 웹캠 `app.py`는 유지된다.

중요: 초기 단일 서버 MediaPipe 구조는 폐기했다. 다른 담당자가 정의한 A→B→C
컨테이너 계약을 기준으로 A는 추론하지 않는다. 8762는 B의 metrics 포트이며,
캔버스는 충돌을 피하기 위해 별도 8770 서비스를 사용한다.

```text
휴대폰 capture.html → ngrok HTTPS/WSS → A(remote_app.py)
→ B vision-analysis:8760 → C pattern-command:8761
→ canvas:8770 → A 모니터 index.html
```

## 주요 파일

- `containers/0-web/app.py`: A FastAPI 웹 게이트웨이, JPEG→raw BGR 변환 및 B 전달
- `containers/0-web/web/capture.html`: 후면 카메라, 640×480 JPEG, 기본 10FPS 송신
- `containers/0-web/web/index.html`: QR, 원본, 캔버스, 명령/줌/추론/FPS 표시
- `containers/1-canvas/app.py`: C 명령 WebSocket 수신 및 모니터 결과 전송
- `containers/1-canvas/drawing_canvas.py`: OpenCV DrawingCanvas와 로컬 웹캠 앱
- `containers/3-pattern-command/`: 실제 제스처 판정 구현
- `start_remote.ps1`: Docker Compose 전체 서비스와 ngrok 시작 및 종료
- `app.py`: 기존 PC 웹캠/OpenCV 앱

제스처 판정은 C가 담당한다. 캔버스는 C가 확정한 명령을 실행할 뿐 제스처를 다시
판단하지 않는다. 따라서 줌 방향 고정과 검지 3프레임 해제 규칙의 단일 책임은 C다.

## 컨테이너 계약

### A → B

주소: `ws://vision-analysis:8760/ingest/{session_id}`

프레임마다 WebSocket 메시지 두 개를 순서대로 보낸다.

1. TEXT: JSON 헤더
2. BINARY: 연속 메모리 raw BGR `uint8` 픽셀

현재 A 헤더 필드: `schema_version`, `session_id`, `frame_id`, `seq`,
`captured_at_ms`, `width`, `height`, `channels`, `dtype`, `color_order`,
`byte_length`.

### B → C

`PATTERN_COMMAND_WS_URL` 기본값은 `ws://pattern-command:8761/landmarks`다.
B 담당 구현이 PRD 6.1 JSON 패킷과 재연결 백오프를 소유한다. 이 작업에서는 B/C
코드를 만들거나 덮어쓰지 않았다.

### C → Canvas

제안 주소는 `ws://canvas:8770/commands/{session_id}`다. 캔버스가 사용하는 필드는
`command`, `mode`, `frame_id`, `seq`, `index_tip`, 선택적인 `index_direction`,
`inference_ms`다. `index_tip`은 0~1 정규화 좌표를 권장한다.

주의: C 담당자의 실제 출력 URL/PRD 명칭이 다르면 `CANVAS_WS_URL` 및
`containers/1-canvas/app.py`의 입력 어댑터를 병합 시 맞춰야 한다.

## 실행

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
$env:NGROK_PATH = "C:\path\to\ngrok.exe"
.\start_remote.ps1
```

출력된 `Monitor` 주소를 PC에서 열고 QR을 휴대폰으로 스캔한다. 휴대폰에서
`카메라 시작`을 누르면 원본/측정 영역과 캔버스가 모니터에 나타난다.

수동 실행은 다음과 같다. B/C 담당 변경이 병합된 뒤 사용한다.

```powershell
$env:SESSION_TOKEN = "hand-board"
$env:PUBLIC_BASE_URL = "https://실제-ngrok-주소.ngrok-free.app"
docker compose up -d --build
ngrok http 8000
```

## 다음 컨텍스트에서 우선 확인할 것

1. 다른 담당자의 최신 B/C 변경이 현재 체크아웃에 들어왔는지 먼저 확인
2. PRD 6.1 실제 JSON 필드와 canvas 입력 어댑터 확정
3. 전체 Docker 빌드 및 8000/8760/8762/8770 health 확인
4. ngrok 실제 HTTPS 주소에서 휴대폰 카메라 권한 확인
5. 휴대폰 실제 손 영상에서 FPS, `inference_ms`, 명령 안정성 확인
6. 휴대폰 회전과 BGR/RGB 설정, 정규화 좌표 방향 확인
7. A→B에도 latest-first 제한 큐가 필요한지 부하 테스트

## 알려진 제한

- 현재 웹 설정은 단일 공개 토큰을 기본으로 한다.
- A는 B에 순차 전달하므로 B가 느리면 브라우저 WebSocket 백프레셔가 생길 수 있다.
- `start_remote.ps1`은 ngrok 주소를 찾지만 이미 시작된 서버의 `PUBLIC_BASE_URL`을
  갱신하지 않는다. 모니터가 ngrok 주소 자체로 열리면 `location.origin`을 사용하므로
  QR은 정상 생성된다.
- 이 체크아웃에는 사용자가 언급한 B/C 최신 구현이 아직 보이지 않았다. 병합 전
  `docker-compose.yml` 충돌을 반드시 검토한다.
- 실제 휴대폰/ngrok 종단 테스트는 사용자 환경에서 확인해야 한다.
