# PRD — 실시간 손동작 인식 원격 그림판 (MediaPipe 적용)

> **Container B (영상 분석 엔진) 제품 요구사항 정의서**
> 채택 모델: **MediaPipe Hand Landmarker (Tasks API)**

| 항목 | 내용 |
| :--- | :--- |
| 문서 버전 | v0.2 — MediaPipe 적용판 |
| 담당 모듈 | **Container B — 영상 분석 엔진** |
| 채택 모델 | MediaPipe Hand Landmarker (`hand_landmarker.task`) |
| 실행 모드 | `LIVE_STREAM` |
| 연관 모듈 | Container A (웹/보안 서버), Container C (동작 인식 엔진) |
| 상태 | 검토 중 |

> ⚠️ **버전 확인 필요**
> MediaPipe Tasks API는 릴리스마다 파라미터명과 모델 번들 경로가 변경된 이력이 있습니다. 본 문서의 API 시그니처는 구현 착수 시점의 공식 문서로 대조 검증한 뒤 확정하세요.

---

## 목차

1. [모델 선정](#1-모델-선정)
2. [채택 모델 상세 스펙](#2-채택-모델-상세-스펙)
3. [Container B 처리 파이프라인](#3-container-b-처리-파이프라인)
4. [모델 설정 파라미터](#4-모델-설정-파라미터)
5. [기능 요구사항](#5-기능-요구사항)
6. [인터페이스 계약](#6-인터페이스-계약)
7. [비기능 요구사항](#7-비기능-요구사항)
8. [MediaPipe 특화 정확도 전략](#8-mediapipe-특화-정확도-전략)
9. [테스트 및 검증 계획](#9-테스트-및-검증-계획)
10. [개발 마일스톤](#10-개발-마일스톤)
11. [리스크와 대응](#11-리스크와-대응)
12. [부록: 참조 구현](#12-부록-참조-구현)

---

## 1. 모델 선정

### 1.1 후보 비교

| 모델 | 출력 | 이 프로젝트 적합성 |
| :--- | :--- | :--- |
| **Hand Landmarker** | 21개 랜드마크 + handedness + world landmarks | ✅ **채택** |
| Gesture Recognizer | 위 전부 + 사전정의 제스처 8종 분류 | ❌ 컨테이너 경계 위반 |
| Pose Landmarker | 전신 33개 랜드마크 (손은 손목까지) | ❌ 손가락 정보 없음 |
| Object Detector | 바운딩 박스만 | ❌ 좌표 정밀도 부족 |
| 레거시 `mp.solutions.hands` | 21개 랜드마크 | ⚠️ Deprecated |

### 1.2 채택 근거: 왜 Hand Landmarker인가

**① 아키텍처 경계와 일치한다**

Gesture Recognizer는 랜드마크 추출과 **제스처 분류를 한 모델에 묶은** 번들입니다. 이를 채택하면 Container C의 존재 이유(변화량 계산·규칙 판별)가 사라지고, 3-컨테이너 마이크로서비스 설계가 무너집니다. B는 **"손이 어디 있는가"** 만 답하고, **"무엇을 하려는가"** 는 C가 판단하는 경계를 지켜야 각 모듈을 독립적으로 교체·튜닝할 수 있습니다.

**② 사전정의 제스처가 요구사항과 맞지 않는다**

Gesture Recognizer의 기본 제스처(`Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Victory`, `Thumb_Up`, `Thumb_Down`, `ILoveYou`, `None`)를 이 프로젝트의 4대 동작에 매핑하면:

| 요구 동작 | 기본 제스처 매핑 | 문제 |
| :--- | :--- | :--- |
| 그리기 (검지) | `Pointing_Up` | **손끝 좌표를 별도로 알아야 그릴 수 있음** — 분류만으로 불충분 |
| 지우기 (검지+중지) | `Victory` | 지우개 위치 좌표가 여전히 필요 |
| Zoom In (손 펼침) | `Open_Palm` | **줌은 이산 분류가 아닌 연속 배율값**이 필요 |
| Zoom Out (손 오므림) | `Closed_Fist` | 위와 동일 |

결국 4개 동작 모두 랜드마크 좌표를 필요로 합니다. 분류 결과는 부가 정보에 그치면서 추론 비용만 늘어납니다.

**③ `hand_world_landmarks`가 핵심 난제를 해결한다**

Hand Landmarker는 두 종류의 좌표를 반환합니다.

| 출력 | 좌표계 | 용도 |
| :--- | :--- | :--- |
| `hand_landmarks` | 이미지 정규화 `[0,1]`, 원점 좌상단 | **화면상 위치** — 그리기 커서 좌표 |
| `hand_world_landmarks` | 미터 단위 3D, 원점은 손의 기하학적 중심 | **손 자체의 형태** — 손가락 벌어짐 정도 |

world landmarks는 카메라와의 거리에 **불변**입니다. Zoom 제스처의 "점간 거리" 판정에서 사용자가 손을 앞뒤로 움직여도 오작동하지 않게 하는 결정적 수단입니다. Gesture Recognizer로는 이 원시 데이터에 접근하되 분류 오버헤드를 함께 떠안게 됩니다.

> **결론**
> **Hand Landmarker를 채택**하고, 제스처 판별 로직은 Container C가 좌표 기반으로 직접 구현한다.

---

## 2. 채택 모델 상세 스펙

### 2.1 모델 번들 구성

`hand_landmarker.task` 번들은 **2단 캐스케이드** 구조입니다.

```
[입력 프레임]
      │
      ▼
┌──────────────────────┐
│ ① Palm Detection     │  손바닥 영역 검출 (BlazePalm 계열)
│    입력 192×192      │  → 손 바운딩 박스 + 회전각
└──────────────────────┘
      │  (첫 프레임 또는 트래킹 실패 시에만 실행)
      ▼
┌──────────────────────┐
│ ② Hand Landmarks     │  21개 랜드마크 회귀
│    입력 224×224      │  → landmarks + world landmarks + presence score
└──────────────────────┘
      │
      └──▶ 다음 프레임의 ROI로 재사용 (트래킹 모드)
```

> 💡 **중요: ROI 트래킹은 이미 내장되어 있다**
> 랜드마크 검출에 성공하면 MediaPipe는 그 결과로 다음 프레임의 관심 영역을 직접 계산해 팜 검출을 건너뜁니다. **별도의 ROI 크롭 로직을 직접 구현하지 마세요.** 우리가 제어하는 것은 언제 팜 검출로 되돌아갈지를 결정하는 신뢰도 임계값뿐입니다.

### 2.2 모델 산출물

| 필드 | 타입 | 설명 |
| :--- | :--- | :--- |
| `hand_landmarks` | `List[List[NormalizedLandmark]]` | x, y ∈ `[0,1]`, z는 손목 기준 상대 깊이 (근사값) |
| `hand_world_landmarks` | `List[List[Landmark]]` | 미터 단위 3D, 원점 = 손 중심 |
| `handedness` | `List[List[Category]]` | `Left` / `Right` + 신뢰도 점수 |

### 2.3 랜드마크 인덱스

| 인덱스 | 부위 | 인덱스 | 부위 |
| :--- | :--- | :--- | :--- |
| 0 | WRIST | 9~12 | MIDDLE MCP / PIP / DIP / TIP |
| 1~4 | THUMB CMC / MCP / IP / TIP | 13~16 | RING MCP / PIP / DIP / TIP |
| 5~8 | INDEX MCP / PIP / DIP / TIP | 17~20 | PINKY MCP / PIP / DIP / TIP |

**본 프로젝트의 주요 사용 인덱스**

| 인덱스 | 용도 |
| :--- | :--- |
| `8` (INDEX_TIP) | 그리기 커서 좌표 |
| `12` (MIDDLE_TIP) | 지우개 모드 판별 |
| `4` (THUMB_TIP) | 핀치 거리 계산 |
| `0`, `9` (WRIST, MIDDLE_MCP) | **hand_scale 기준선** |

### 2.4 런타임 환경

| 항목 | 선택 | 비고 |
| :--- | :--- | :--- |
| 언어 | Python 3.10+ | `pip install mediapipe` |
| 실행 모드 | `LIVE_STREAM` | 비동기 콜백 + 내부 트래킹 활성화 |
| 델리게이트 | **CPU (XNNPACK)** | Linux Python GPU 델리게이트는 지원이 제한적 — 벤치마크 후 결정 |
| 모델 파일 | `hand_landmarker.task` (float16) | 이미지 빌드 시 번들에 포함, 런타임 다운로드 금지 |

> 모델 파일은 Docker 이미지 빌드 단계에서 내려받아 포함시킵니다. 컨테이너 기동 시 외부 다운로드에 의존하면 네트워크 장애가 곧 서비스 장애가 됩니다.

---

## 3. Container B 처리 파이프라인

```
[Container A: 프레임 수신]
      │
      ▼
① 색공간 정규화        YUV420 / BGR → RGB (uint8, C-contiguous)
      │
      ▼
② 기하 보정            rotation 적용 + 전면카메라 미러링 해제
      │
      ▼
③ 리사이즈             종횡비 유지 letterbox → 640×480
      │
      ▼
④ mp.Image 생성        ImageFormat.SRGB
      │
      ▼
⑤ detect_async()       단조증가 타임스탬프(ms) 필수
      │
      ▼ (콜백)
⑥ 결과 수신            landmarks + world_landmarks + handedness
      │
      ▼
⑦ 좌표 역매핑          letterbox 오프셋 제거 → 원본 프레임 기준 [0,1]
      │
      ▼
⑧ hand_scale 산출      dist(LM0, LM9) 계산
      │
      ▼
⑨ One Euro Filter      21개 랜드마크 시간축 스무딩
      │
      ▼
⑩ 이상치 검사          프레임 간 이동량 임계치 초과 시 폐기
      │
      ▼
⑪ 패킷 직렬화          → Container C 전송
```

---

## 4. 모델 설정 파라미터

| 파라미터 | 기본값 | **권장값** | 근거 |
| :--- | :--- | :--- | :--- |
| `running_mode` | `IMAGE` | **`LIVE_STREAM`** | 내부 트래킹 활성화 + 비동기 처리로 블로킹 제거 |
| `num_hands` | 1 | **1** | 배경 오검출 억제, 추론 비용 최소화 |
| `min_hand_detection_confidence` | 0.5 | **0.7** | 팜 검출 단계 임계값. 높이면 배경의 손 모양 오검출 감소 |
| `min_hand_presence_confidence` | 0.5 | **0.6** | 낮으면 손이 사라져도 이전 좌표를 붙들고 늘어짐 |
| `min_tracking_confidence` | 0.5 | **0.5** | 높이면 팜 재검출이 잦아져 프레임 시간이 튐 |
| `delegate` | `CPU` | **`CPU`** | 벤치마크 결과에 따라 재검토 |

### 4.1 파라미터 튜닝 원칙

세 임계값은 **서로 상충**하므로 개별 최적화가 아닌 조합으로 튜닝해야 합니다.

| 증상 | 원인 추정 | 조정 방향 |
| :--- | :--- | :--- |
| 손을 치웠는데 선이 계속 그려짐 | presence 임계값이 낮음 | `min_hand_presence_confidence` ↑ |
| 손이 있는데 자주 끊김 | detection 임계값이 높음 | `min_hand_detection_confidence` ↓ |
| 프레임 처리 시간이 들쭉날쭉 | 팜 재검출이 빈번함 | `min_tracking_confidence` ↓ |
| 배경 물체를 손으로 오인 | detection 임계값이 낮음 | `min_hand_detection_confidence` ↑ |

---

## 5. 기능 요구사항

### 5.1 전처리

| ID | 요구사항 | 우선순위 |
| :--- | :--- | :--- |
| FR-B-01 | 입력 프레임을 **RGB uint8 연속 배열**로 변환한다 (MediaPipe는 BGR 미지원) | P0 |
| FR-B-02 | `rotation`과 `mirrored` 플래그를 추론 **이전에** 보정한다 | P0 |
| FR-B-03 | 종횡비를 유지한 letterbox 리사이즈로 640×480에 맞춘다 | P0 |
| FR-B-04 | 처리 지연 시 대기 프레임을 폐기하고 최신 프레임만 처리한다 | P0 |
| FR-B-05 | 저조도 프레임에 CLAHE 기반 대비 보정을 적용한다 | P1 |

### 5.2 추론

| ID | 요구사항 | 우선순위 |
| :--- | :--- | :--- |
| FR-B-06 | `HandLandmarker`를 `LIVE_STREAM` 모드로 초기화하고 콜백으로 결과를 수신한다 | P0 |
| FR-B-07 | `detect_async()`에 전달하는 타임스탬프는 **단조증가**를 보장한다 (위반 시 예외 발생) | P0 |
| FR-B-08 | `hand_landmarks`, `hand_world_landmarks`, `handedness`를 모두 수집한다 | P0 |
| FR-B-09 | `HandLandmarker` 인스턴스는 세션 단위로 생성하고 종료 시 `close()`로 해제한다 | P0 |
| FR-B-10 | ~~자체 ROI 크롭 구현~~ → **MediaPipe 내장 트래킹에 위임** (2.1 참조) | — |

### 5.3 후처리

| ID | 요구사항 | 우선순위 |
| :--- | :--- | :--- |
| FR-B-11 | letterbox 패딩 오프셋을 제거해 **원본 프레임 기준 정규화 좌표**로 역매핑한다 | P0 |
| FR-B-12 | `hand_scale = dist(LM0, LM9)`를 산출해 패킷에 동봉한다 | P0 |
| FR-B-13 | 21개 랜드마크에 One Euro Filter를 적용한다 (**MediaPipe는 내부 스무딩을 제공하지 않음**) | P0 |
| FR-B-14 | 프레임 간 이동량이 `hand_scale`의 일정 배수를 초과하면 이상치로 폐기한다 | P1 |
| FR-B-15 | 랜드마크가 프레임 경계에 근접하면 `near_edge: true`를 표기한다 | P1 |
| FR-B-16 | 손 미검출 시 `hand_present: false`, `landmarks: null`로 전송한다 (`(0,0)` 금지) | P0 |

### 5.4 관측성

| ID | 요구사항 | 우선순위 |
| :--- | :--- | :--- |
| FR-B-17 | 추론 시간, 팜 재검출 빈도, 검출률을 지표로 노출한다 | P1 |
| FR-B-18 | 디버그 모드에서 랜드마크 오버레이 영상을 출력한다 | P2 |

---

## 6. 인터페이스 계약

### 6.1 출력 스키마 (B → C)

```json
{
  "session_id": "a1b2c3d4",
  "seq": 10421,
  "capture_ts": 1735891234567,
  "processed_ts": 1735891234589,
  "hand_present": true,
  "handedness": { "label": "Right", "score": 0.98 },
  "frame": { "w": 640, "h": 480 },

  "landmarks": [
    { "x": 0.412, "y": 0.633, "z": -0.021 },
    { "x": 0.428, "y": 0.601, "z": -0.018 }
  ],

  "world_landmarks": [
    { "x": -0.031, "y": 0.042, "z": 0.008 },
    { "x": -0.024, "y": 0.037, "z": 0.006 }
  ],

  "hand_scale": 0.187,

  "quality": {
    "near_edge": false,
    "filtered": true,
    "outlier_dropped": false
  }
}
```

### 6.2 계약 규칙

> 아래 6개 항목은 **API 스펙에 명문화**합니다. 실제 버그의 대부분이 이 지점의 해석 차이에서 발생합니다.

1. **좌표계** — 원점 좌상단, x는 오른쪽, y는 **아래쪽**이 양의 방향 (MediaPipe 기본값과 동일).
2. **정규화** — `landmarks`는 `[0,1]`. 프레임 밖 랜드마크는 음수나 1 초과가 될 수 있으며 **클램핑하지 않는다**.
3. **미러링** — B가 이미 보정해 출력한다. C는 추가 반전을 하지 않는다.
4. **단위 구분** — `landmarks`는 화면 좌표(그리기용), `world_landmarks`는 미터(형태 판별용). **혼용 금지.**
5. **결측** — `hand_present: false`일 때 `landmarks`와 `world_landmarks`는 모두 `null`.
6. **시간 기준** — 변화량 계산의 기준은 `capture_ts`. `processed_ts`는 관측 전용.

### 6.3 Container C를 위한 사용 지침

| C의 판별 항목 | 사용할 데이터 | 이유 |
| :--- | :--- | :--- |
| 그리기 커서 위치 | `landmarks[8]` (x, y) | 화면 좌표가 그대로 캔버스에 매핑됨 |
| 손가락 폄/굽힘 | `world_landmarks` | 카메라 거리 무관 |
| Zoom 배율 | `world_landmarks` 점간 거리 | **원근 왜곡 제거된 유일한 수단** |
| 이동 속도 | `landmarks` Δ ÷ `capture_ts` Δ | 시간 정규화 필수 |

---

## 7. 비기능 요구사항

| ID | 지표 | 목표값 |
| :--- | :--- | :--- |
| NFR-01 | 프레임당 추론 시간 (CPU) | p50 ≤ 15 ms, p95 ≤ 25 ms |
| NFR-02 | 안정 처리 프레임레이트 | **30 fps 균일** (불균일한 60 fps보다 우선) |
| NFR-03 | End-to-End 지연 (촬영 → 캔버스) | p95 ≤ 150 ms |
| NFR-04 | 출력 패킷 크기 | ≤ 2 KB/frame (world landmarks 포함) |
| NFR-05 | 손 검출률 | ≥ 98% (표준 조명) / ≥ 90% (저조도) |
| NFR-06 | 정지 상태 지터 (LM 8) | 표준편차 ≤ 1.5 px (640×480 환산) |
| NFR-07 | 오검출률 (손 없는 프레임) | ≤ 1% |
| NFR-08 | 팜 재검출 발생률 | ≤ 5% of frames |

> **NFR-08 보충** — 팜 재검출이 발생한 프레임은 추론 시간이 2~3배로 튑니다. 이 비율이 높으면 평균 FPS는 정상이어도 체감 끊김이 발생하므로 별도 지표로 관리합니다.

---

## 8. MediaPipe 특화 정확도 전략

### 8.1 [최우선] world_landmarks로 거리 불변성 확보

이 프로젝트의 최대 함정은 **손과 카메라의 거리 변화**입니다. 이미지 좌표로 "엄지-검지 거리"를 재면, 사용자가 손을 앞으로 내미는 것만으로 Zoom In이 오발동합니다.

```python
# ❌ 원근 왜곡에 취약
d = dist(landmarks[4], landmarks[8])

# ✅ 카메라 거리 불변 (미터 단위)
d = dist(world_landmarks[4], world_landmarks[8])

# ✅ 차선책: 이미지 좌표를 hand_scale로 정규화
d = dist(landmarks[4], landmarks[8]) / dist(landmarks[0], landmarks[9])
```

world landmarks를 쓸 수 없는 상황(값이 불안정하거나 라이브러리 버전 이슈)에 대비해 `hand_scale`도 함께 제공하여 C가 두 방식 중 선택할 수 있게 합니다.

### 8.2 [최우선] One Euro Filter 자체 구현

MediaPipe Hand Landmarker는 **랜드마크 스무딩 옵션을 제공하지 않습니다.** 원시 출력은 손을 완전히 정지시켜도 프레임당 1~3 px 떨리며, 이를 그대로 그리면 선이 톱니 모양이 됩니다.

| 필터 | 특성 | 권장도 |
| :--- | :--- | :--- |
| **One Euro Filter** | 저속엔 강한 스무딩, 고속엔 최소 지연 | ⭐ 실시간 드로잉 표준 |
| EMA | 구현 간단, 빠른 동작에서 커서가 뒤처짐 | 프로토타입용 |
| Kalman | 이론적 우수, 튜닝 비용 큼 | 선택적 |

튜닝 대상 파라미터:

| 파라미터 | 역할 | 조정 방향 |
| :--- | :--- | :--- |
| `min_cutoff` | 기본 스무딩 강도 | 낮출수록 매끄럽지만 지연 증가 |
| `beta` | 속도 반응성 | 높일수록 빠른 동작에서 지연 감소 |

> 필터는 **각 랜드마크의 x, y에 독립적으로** 적용합니다. 손이 사라졌다 다시 나타나면 필터 상태를 초기화해야 이전 위치에서 끌려오는 현상을 막을 수 있습니다.

### 8.3 MediaPipe 고유의 함정

| 함정 | 증상 | 대응 |
| :--- | :--- | :--- |
| **BGR 입력** | 검출률이 통째로 급락 | `cv2` 사용 시 `cv2.COLOR_BGR2RGB` 필수 |
| **비단조 타임스탬프** | 런타임 예외로 프로세스 중단 | 내부 카운터로 단조증가 보장 |
| **비연속 배열** | `mp.Image` 생성 실패 | `np.ascontiguousarray()` 적용 |
| **`z` 값 오용** | 깊이 판정이 부정확 | `landmarks[].z`는 근사값 — 절대 깊이엔 `world_landmarks` 사용 |
| **인스턴스 공유** | 세션 간 트래킹 상태 오염 | 세션당 독립 인스턴스, 종료 시 `close()` |
| **과대 해상도 입력** | 처리 시간만 증가 | 모델 입력이 224×224이므로 640×480 이상은 무의미 |

### 8.4 입력 품질 확보

모델 파라미터를 만지기 전에 확인해야 할 항목들입니다. 정확도 문제의 상당수가 여기서 발생합니다.

| 항목 | 문제 | 대응 |
| :--- | :--- | :--- |
| 손 크기 | 프레임의 20% 미만이면 정확도 저하 | 사용자에게 적정 거리 안내 UI |
| 모션 블러 | 빠른 동작에서 손끝이 뭉개짐 | 노출 시간 단축, 조명 확보 |
| 저조도 | 팜 검출 실패 | CLAHE 대비 보정 |
| 압축 아티팩트 | 손가락 경계 손실 | WebRTC 비트레이트 하한 설정 |
| 프레임 경계 | 손이 잘리면 랜드마크가 외삽되어 튐 | `near_edge` 플래그로 C에 통지 |

### 8.5 지연이 곧 정확도다

큐에 프레임이 쌓이면 좌표는 "정확하지만 늦은" 값이 되고, 사용자는 이를 부정확하다고 인식합니다.

- `detect_async()` 콜백이 아직 오지 않았는데 새 프레임이 도착하면 **대기 프레임을 폐기**합니다.
- `capture_ts`와 `processed_ts` 차이를 상시 모니터링해 지연 누적을 조기 감지합니다.
- 입력이 60 fps여도 **30 fps로 균일하게 다운샘플링**합니다. Container C의 변화량 계산은 프레임 간격이 일정할 때만 신뢰할 수 있으며, 불규칙하게 드롭된 60 fps는 오히려 속도 오판을 유발합니다.

---

## 9. 테스트 및 검증 계획

### 9.1 회귀 테스트 영상 세트

**"정확도를 올렸다"고 말하려면 먼저 측정해야 합니다.** 조건별 영상을 고정 녹화해두고 코드 변경마다 동일 입력으로 지표를 재산출합니다.

| 축 | 조건 |
| :--- | :--- |
| 조명 | 밝은 실내 / 어두운 실내 / 역광 |
| 배경 | 단색 벽 / 복잡한 배경 / 사람이 지나가는 배경 |
| 거리 | 30 cm / 60 cm / 100 cm |
| 대상 | 다양한 손 크기·피부톤 (최소 5인) |
| 동작 | 4대 제스처 각 10회 + 정지 10초 |

### 9.2 핵심 측정 지표

| 지표 | 측정 방법 | 비고 |
| :--- | :--- | :--- |
| **정지 지터** | 손 고정 10초, LM 8의 x·y 표준편차 | 가장 저렴하고 효과적인 회귀 지표 |
| **거리 불변성** | 동일 제스처를 30/60/100 cm에서 수행, world 좌표 점간 거리 편차 | **8.1 검증의 핵심** |
| 검출률 | 손이 있는 프레임 중 `hand_present: true` 비율 | 조명 조건별 분리 측정 |
| 오검출률 | 손 없는 영상에서의 검출 비율 | |
| 팜 재검출률 | 전체 프레임 대비 재검출 발생 비율 | NFR-08 |
| E2E 지연 | 화면에 타임코드 표시 후 캔버스 반영 시점 촬영 | 실측 필수 |

### 9.3 단위 테스트 필수 항목

- [ ] RGB/BGR 변환 정합성 (뒤바뀜 검출)
- [ ] 미러링·회전 보정 후 좌표 대칭성
- [ ] letterbox 역매핑 정확도 (알려진 좌표 왕복 검증)
- [ ] 타임스탬프 단조증가 보장
- [ ] `hand_present: false`일 때 `landmarks`가 `null`인지
- [ ] 세션 종료 후 필터 상태 초기화 여부
- [ ] 이상치 폐기 임계값 경계 동작

---

## 10. 개발 마일스톤

- [ ] **B-1 (환경)** — `mediapipe` 설치, `hand_landmarker.task` 번들 확보, Docker 이미지 빌드
      → *완료 기준: 정지 이미지 1장에서 21개 랜드마크 추출 성공*
- [ ] **B-2 (스트림)** — `LIVE_STREAM` 모드 + 비동기 콜백 파이프라인 구성
      → *완료 기준: 로컬 영상 파일 30 fps 연속 처리, 타임스탬프 예외 없음*
- [ ] **B-3 (전처리)** — 색공간·회전·미러링·letterbox 및 역매핑 구현
      → *완료 기준: 단위 테스트 9.3 전 항목 통과*
- [ ] **B-4 (계약)** — 정규화 좌표 + world landmarks + `hand_scale` 패킷 출력
      → *완료 기준: Container C와 스키마 연동 성공*
- [ ] **B-5 (안정화)** — One Euro Filter, 이상치 제거, 파라미터 튜닝
      → *완료 기준: NFR-06 지터 목표 달성*
- [ ] **B-6 (검증)** — 회귀 테스트 세트 구축 및 지표 자동 측정
      → *완료 기준: 전 지표 대시보드화, 거리 불변성 검증 통과*

---

## 11. 리스크와 대응

| 리스크 | 영향 | 대응 |
| :--- | :--- | :--- |
| 손 거리 변화로 Zoom 오발동 | **높음** | `world_landmarks` 우선 사용 + `hand_scale` 병행 제공 (8.1) |
| 스무딩 과다로 드로잉 반응 지연 | **높음** | 실사용 영상 기반 One Euro 파라미터 튜닝 |
| 처리 지연 누적으로 체감 정확도 하락 | **높음** | latest-frame-wins + 지연 상시 모니터링 |
| MediaPipe 버전 업그레이드 시 API 변경 | 중간 | 버전 핀 고정, 래퍼 계층으로 격리 |
| Python GPU 델리게이트 미지원 | 중간 | CPU 벤치마크 선행, 미달 시 프레임레이트 하향 |
| 저조도·역광에서 팜 검출 실패 | 중간 | 대비 보정 + 조명 안내 UI |
| B/C 간 좌표계 해석 불일치 | 중간 | 6.2 계약 규칙 문서화 + 통합 테스트 |

---

## 12. 부록: 참조 구현

> 아래 코드는 구조 참고용 스켈레톤입니다. API 시그니처는 사용 버전의 공식 문서로 검증하세요.

### 12.1 초기화 및 추론

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode


def on_result(result, output_image, timestamp_ms: int):
    """추론 완료 콜백. 여기서 후처리 → 패킷 전송."""
    if not result.hand_landmarks:
        emit_packet(hand_present=False, capture_ts=timestamp_ms)
        return

    lms = result.hand_landmarks[0]
    world = result.hand_world_landmarks[0]
    handed = result.handedness[0][0]

    lms = unletterbox(lms)          # 패딩 오프셋 제거
    lms = euro_filter.apply(lms, timestamp_ms)
    scale = hand_scale(lms)

    emit_packet(
        hand_present=True,
        landmarks=lms,
        world_landmarks=world,
        handedness={"label": handed.category_name, "score": handed.score},
        hand_scale=scale,
        capture_ts=timestamp_ms,
    )


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)

landmarker = HandLandmarker.create_from_options(options)
```

### 12.2 프레임 투입

```python
import cv2
import numpy as np

_last_ts = -1

def process_frame(bgr_frame, capture_ts_ms: int):
    global _last_ts

    # 타임스탬프 단조증가 보장 (위반 시 MediaPipe 예외 발생)
    if capture_ts_ms <= _last_ts:
        capture_ts_ms = _last_ts + 1
    _last_ts = capture_ts_ms

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)   # BGR 입력 금지
    rgb = letterbox(rgb, (640, 480))                   # 종횡비 유지
    rgb = np.ascontiguousarray(rgb)                    # 연속 배열 필수

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, capture_ts_ms)   # 논블로킹
```

### 12.3 hand_scale 산출

```python
import math

WRIST, MIDDLE_MCP = 0, 9

def hand_scale(landmarks) -> float:
    """손목~중지 MCP 거리. 화면상 손 크기의 기준값."""
    a, b = landmarks[WRIST], landmarks[MIDDLE_MCP]
    return math.hypot(a.x - b.x, a.y - b.y)
```

### 12.4 거리 불변 측정 (Container C 참고용)

```python
THUMB_TIP, INDEX_TIP = 4, 8

def pinch_distance_metric(world_landmarks) -> float:
    """미터 단위. 카메라 거리에 불변."""
    a, b = world_landmarks[THUMB_TIP], world_landmarks[INDEX_TIP]
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
```