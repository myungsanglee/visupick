# VisuPick

> **3D vision-guided bin picking workbench for KUKA robots + Zivid 3D camera**
> Hand-eye calibration · 6D pose estimation · Bin picking · 통합 GUI

`VisuPick`은 산업용 로봇(KUKA KRC5 + KSS 8.7) 환경에서 3D 카메라(Zivid / RealSense / Percipio)로 객체를 인식하고, hand-eye calibration을 거쳐 로봇이 정확히 picking 하도록 돕는 통합 데스크탑 워크벤치입니다. **데이터 수집 → 캘리브레이션 → 검증 → Bin Picking / CAD 매칭 / Surface Tracking** 을 하나의 PySide6 GUI(탭 5개)에서 처리합니다. 검출은 RF-DETR + SAM 3 텍스트 프롬프트, 파지는 진공 그리퍼(SMC ZK2)를 씁니다.

이 프로젝트는 학습 및 연구 용도로 작성되었고, 알고리즘이 어떻게 동작하는지 [docs/](docs/) 폴더에 상세 설명서가 포함되어 있습니다.

---

## 주요 기능

### 1. Hand-Eye Calibration
- **Eye-to-Hand** / **Eye-in-Hand** 두 모드 지원
- 체커보드 + Zivid 3D 포인트 클라우드 기반 정밀 추정 (`solvePnP` fallback 포함)
- OpenCV 5종 알고리즘 자동 비교 (TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS) → 가장 일관성 좋은 결과 선택
- 비선형 정밀화 + greedy outlier 제거 → **일관성 0.2 mm 이하 달성**
- 자세한 원리: [docs/hand_eye_calibration.md](docs/hand_eye_calibration.md)

### 2. Bin Picking (AI 검출 + 3D 포인트클라우드)
- **객체 검출 2경로** — `RF-DETR`(ONNX/TensorRT, 학습된 클래스) + **SAM 3 텍스트 프롬프트**(명사구 하나로 학습 없이 임의 객체 분할, Grounding DINO 불필요). 추론 시간(ms) 표시.
- 검출 → 객체별 **3D 중심 + 표면 법선** 추정. 2D ROI 드래그로 박스 영역만 필터.
- **마스크 → OBB(회전 사각형) → 여는 방향 추정** — 투명 화장품 케이스처럼 CAD 없이 방향을 잡아야 할 때. 방식 3종(이음선 에지 / 내부 밝기 / **내부 격자 비대칭**).
- **Grasp 설정(파지 전략)** — CAD 없이 잡을 때 노이즈 심한 3D 대신 자유도를 고정. "고정 평면 광선 투영 + 수직 접근 + 열림 방향 정렬"의 **4-DOF top-down 파지**(투명·평면 객체용).
- **진공 그리퍼 픽 사이클** — Approach→하강→진공 ON→상승→놓기→진공 OFF(+블로우)→Home 복귀 원버튼 실행. 시퀀스 큐로 다중 픽 배치 실행.
- 객체 중심에 **Tool 좌표축 시각화**(접근 자세 실시간 미리보기).
- 자세한 원리: [docs/bin_picking.md](docs/bin_picking.md)

### 3. CAD 기반 6D Pose Matching
- **FPFH + ICP (RANSAC / FGR)** — 정자세 환경에 빠르고 정확
- **PPF (OpenCV Surface Matching) + Open3D ICP 정밀화** — 무작위 자세 + 부분 가시성에 강건
- **PPF 전체장면(DBSCAN 없이)** — 분할 없이 전체 장면에 voting 1회로 다중 인스턴스 직접 검출(상용/Drost 방식). 쌓임·부분 가림에 강건. 장면 다운샘플·pre-ICP NMS·단계별 시간 계측으로 속도 튜닝.
- (대안) **DBSCAN 클러스터 분리** + **작업대 평면 자동 제거**
- Grasp 위치(X/Y/Z) + 회전(ABC) 3D 설정 → 사용자가 객체별 잡는 자세 정밀 조정
- 인스턴스 클릭 시 3D 뷰에 Tool 좌표축 + Approach 경로 시각화
- 자세한 원리: [docs/cad_matching.md](docs/cad_matching.md)

### 4. Surface Tracking (표면 추적)
- 굴곡진 표면(예: 자동차 외관)에 매직펜으로 그린 **검은 선을 인식** → 그 경로를 로봇 툴이 **표면 법선에 수직 정렬한 채** 따라 이동.
- 검은 선 검출(adaptive threshold + thinning) → 시작/끝점 클릭 → skeleton BFS path → **3D 거리 기준 mm 간격 샘플** → 각 점 법선(국소 평면 피팅)으로 자세 계산 → **offset mm 만큼 표면 바깥으로 띄움**.
- 자세한 원리: [docs/surface_tracking.md](docs/surface_tracking.md)

### 5. 진공 그리퍼 + 공통 로봇 제어
- **SMC ZK2 진공 이젝터 그리퍼** 제어(진공 ON/OFF/블로우) — C3Bridge가 `$OUT` 직접 쓰기를 거부하므로 KRL 인터럽트로 대신 적용(`robo_vac_*` 계약).
- Bin Picking / CAD 매칭 / Surface Tracking 세 탭이 **공통 로봇 제어 Mixin**(단일 이동·시퀀스 큐·Z 안전·AUT 속도 상한·Space 비상정지·픽 사이클) 공유.

### 6. 통신 / KRL
- KUKA `C3Bridge` 프로토콜 (포트 7000, KukaVarProxy 호환)
- 20슬롯 모션 큐 KRL 프로그램 (`krl/ext_move.src`) — PTP / LIN / 비상정지 (`RESUME`) / 안전 일시정지
- AUT 모드 자동 50 % 속도 상한, Z 안전 한계, Space 비상정지 단축키
- 자세한 원리: [docs/kuka_communication.md](docs/kuka_communication.md)

---

## 검증된 하드웨어 / 소프트웨어 환경

| 항목 | 사양 |
|---|---|
| 로봇 | **KUKA KR 10 R1100-2** (KR AGILUS) |
| 컨트롤러 | **KRC5 micro**, KSS 8.7.7 HF1 |
| 그리퍼 | **SMC ZK2 진공 이젝터 + ZSE 압력 스위치** (흡착식). EtherCAT(EK1100+EL2889) 채널로 24V 밸브 제어 → `$OUT[7]` 진공 / `$OUT[8]` 블로우 |
| 카메라 ① | **Zivid 2 M70** (구조광 3D, ~0.1 mm 정밀) |
| 카메라 ② | **Intel RealSense D415** (Active Stereo, ~2 mm 정밀) |
| 카메라 ③ | **Percipio FM815-IX-E1** (GigE 구조광, 선택적) |
| OS (PC) | Ubuntu 24.04 LTS |
| Python | 3.10 이상 |

### 카메라 추상화 + 추가 카메라

본 시스템은 `BaseCamera` 추상 인터페이스 + factory 패턴으로 설계되어 있어 **다른 3D 카메라도 같은 인터페이스로 통합 가능**합니다. 새 카메라 추가 시 [base_camera.py](base_camera.py) 의 추상 메서드들만 구현 → [camera_factory.py:`_REGISTRY`](camera_factory.py) 에 등록하면 GUI 콤보에 자동 노출.

| 카메라 | 본 시스템 권장 calibration 방식 | 이유 |
|---|---|---|
| Zivid (정밀 깊이) | `pointcloud` (3D 직접 매칭) | 깊이 정밀도가 RGB intrinsics 보다 높음 |
| RealSense (저정밀 깊이) | `pnp` (solvePnP) | RGB intrinsics 가 깊이보다 정밀 |
| 불확실하면 | **`compare`** (두 방법 비교) | 일관성 metric 자동 비교 → 더 좋은 쪽 채택 |

다른 KUKA 모델 / 카메라로도 작동 가능 (KSS 8.5+ + C3Bridge 가능한 모든 KRC). 시도해보고 결과를 issue로 공유해주시면 좋겠습니다.

---

## 설치

### Python 환경
```bash
git clone https://github.com/<your-user>/visupick.git
cd visupick
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 카메라 SDK (사용하는 것만)

**Zivid 사용 시:**
시스템 차원의 Zivid SDK가 별도로 필요합니다 (Python 패키지만으로는 작동 안 함).
→ https://support.zivid.com/latest/getting-started/software-installation.html

설치 후 Zivid Studio에서 카메라 설정 YML 파일을 export 하여 **`config/` 폴더**에 두세요 (예: `config/zivid_settings_manufacturing_specular.yml`). 프로그램의 "설정 (YML)" 버튼이 이 폴더를 기본으로 열어줍니다. 샘플 두 개가 리포에 포함되어 있습니다.

**Intel RealSense (D415 등) 사용 시:**
Linux 는 시스템 차원의 `librealsense` SDK 가 필요합니다 (Python 패키지만으로는 카메라 인식 안 됨).
→ https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

(옵션) RealSense Viewer 에서 export 한 JSON 설정도 `config/` 에 두면 프로그램이 동일하게 로드합니다. JSON 없어도 기본 설정으로 동작.

GUI 상단의 **"카메라"** 콤보로 Zivid/RealSense 중 선택 후 "카메라 연결".

### KUKA 컨트롤러 측 설정
1. **C3Bridge 활성화** — KRC 컨트롤러에 `C3Bridge` (또는 KukaVarProxy 호환) 서버를 설치하고 부팅 시 자동 실행되도록 설정.
2. **KRL 프로그램 업로드** — `krl/ext_move.src` 와 `krl/ext_move.dat` 을 컨트롤러로 복사 (예: KRC `R1/Programs/`).
3. **TOOL_DATA / BASE_DATA 측정** — SmartPAD에서 그리퍼 TCP를 측정해 `TOOL_DATA[1]` 에 저장. `$BASE` 는 `$NULLFRAME` 사용 권장.
4. **ext_move 실행** — SmartPAD에서 `ext_move` 선택 후 실행 (LOOP 대기 상태).
5. 외부 PC에서 본 프로그램 실행 → 로봇 IP와 Tool 번호 입력 → "연결".

자세한 통신 메시지 포맷과 KRL 큐 동작은 [docs/kuka_communication.md](docs/kuka_communication.md) 를 참고하세요.

### 객체 검출기 (옵션, Bin Picking 탭 사용 시)

검출기는 [object_detector.py](object_detector.py) 로 카메라처럼 추상화돼 있고, 둘 다 선택적입니다 (없어도 앱은 정상 시작, 해당 버튼만 안내 에러):

**① RF-DETR** — ONNX / TensorRT 래퍼 `detector.py` (별도 리포). 학습된 클래스만 검출.
```bash
export RFDETR_DETECTOR_DIR=/path/to/rf-detr/tmp      # detector.py 가 있는 디렉터리
export RFDETR_MODEL_PATH=/path/to/rfdetr-nano.engine # TensorRT 엔진(.engine/.onnx)
```
필요 패키지: `supervision`, `tensorrt`, `pycuda`(TensorRT) 또는 `onnxruntime-gpu`(ONNX).
클래스 매핑은 `bin_picking_tab.py` 의 `RFDETR_CLASSES`.

**② SAM 3 (텍스트 프롬프트)** — Meta 공식 repo(facebookresearch/sam3). 명사구 하나(예: `cosmetic case`)로 **학습 없이** 임의 객체를 분할. 무거움(수억 파라미터, GPU 권장). gated repo라 `hf auth login` + 접근 승인 필요.
```bash
export SAM3_MODEL_DIR=/path/to/sam3   # repo 가 pip 경로에 없을 때
```
> ⚠️ RF-DETR(TensorRT/pycuda)과 SAM3(PyTorch)를 한 프로세스에서 번갈아 쓰면 CUDA 컨텍스트 충돌이 날 수 있어, `detector.py` 는 pycuda 가 PyTorch 의 primary CUDA 컨텍스트를 공유하도록 처리했다. (자세히는 커밋 히스토리 참고)

---

## 실행

```bash
python main.py
```

GUI가 열리면 다음 순서로 진행:

1. **카메라 연결** + **카메라 설정 (YML)** 로드
2. **로봇 연결** (IP, Tool 번호 입력)
3. 작업 시나리오 선택 (탭 5개):
   - **데이터 수집** 탭 → 캘리브레이션용 포즈 수집
   - **검증** 탭 → 캘리브레이션 정확도 확인
   - **Bin Picking** 탭 → RF-DETR/SAM3 검출 + 3D 포즈 + 진공 픽킹
   - **CAD 매칭** 탭 → CAD 6D pose(PPF 전체장면 등) + 픽킹
   - **Surface Tracking** 탭 → 표면 위 검은 선 따라 법선 정렬 이동

> **이미지 저장 버튼 2개** (상단 연결 영역) — 둘 다 **현재 활성 탭** 기준으로 동작하고, 저장 시 파일 이름을 지정할 수 있다 (`data/debug_captures/`, 무손실 PNG):
> - **💾 원본 저장** — 캡처된 **원본 이미지**(오버레이 없음). 알고리즘을 앱 없이 오프라인에서 튜닝·재현할 때.
> - **🖼 렌더링 저장** — 현재 **보이는 뷰를 그대로**. 결과 보고·문서용 스크린샷에.
>   - **2D 뷰**일 때: 검출 bbox·마스크·OBB·여는 방향 화살표·ROI 등 오버레이 포함
>   - **3D 뷰**일 때: 포인트클라우드·Bin Box·Tool 좌표축 등 **현재 카메라 시점 스크린샷** (CAD 매칭의 CAD 뷰·클러스터 뷰도 동일)

### 단축키 (탭에 따라 다름)
| 키 | 동작 |
|:-:|---|
| `C` | 캡처 (데이터 수집 탭) |
| `S` | 포즈 저장 (데이터 수집 탭) |
| `Space` | 비상정지 (Bin Picking / CAD 매칭 탭) |

---

## 프로젝트 구조

```
visupick/
├── main.py                      # 진입점 + 메인 윈도우 + 데이터 수집/검증 탭
├── bin_picking_tab.py           # Bin Picking 탭 (RF-DETR/SAM3 + 3D + Grasp 설정)
├── cad_matching_tab.py          # CAD 6D 매칭 탭 (FPFH/PPF/전체장면 PPF)
├── surface_tracking_tab.py      # Surface Tracking 탭 (검은 선 따라 법선 정렬 이동)
├── robot_control_mixin.py       # 세 탭 공통: 로봇 이동·시퀀스 큐·안전·진공·픽 사이클·E-stop
├── object_detector.py           # 검출기 추상화: RFDetrDetector / Sam3Detector
├── opening_analysis.py          # 순수 CV: OBB + 여는 방향(이음선/밝기/격자)
├── image_view.py                # 2D 이미지 라벨 (Zoomable/Draggable/ClickPoint)
├── pointcloud_view.py           # 3D 포인트클라우드 뷰 (PyVista, 세 탭 공유)
├── calibration.py               # Hand-eye 알고리즘 + 객체 자세 계산
├── kuka_robot.py                # C3Bridge 통신 + 큐 기반 로봇 제어 + 진공
├── base_camera.py               # 카메라 추상 인터페이스 (BaseCamera)
├── camera_factory.py            # 카메라 종류 → 인스턴스 생성 (지연 import)
├── zivid_camera.py              # Zivid SDK 래퍼
├── realsense_camera.py          # Intel RealSense SDK 래퍼 (D415 검증)
├── percipio_camera.py           # Percipio GigE SDK 래퍼 (선택적)
├── krl/
│   ├── ext_move.src             # KRL 모션 큐 프로그램 (모션 + 진공 인터럽트)
│   └── ext_move.dat             # DEFDAT
├── config/                      # Zivid 카메라 설정 (YML, Zivid Studio에서 export)
├── cad_models/                  # 매칭에 사용할 CAD 파일 (STL/OBJ/PLY)
├── docs/
│   ├── hand_eye_calibration.md  # 캘리브레이션 알고리즘 학습 문서
│   ├── bin_picking.md           # 빈 픽킹(검출·OBB·여는방향·Grasp·픽사이클) 학습 문서
│   ├── cad_matching.md          # CAD 6D pose 매칭(PPF 전체장면 등) 학습 문서
│   ├── surface_tracking.md      # 표면 추적(검은 선 → 법선 정렬 이동) 학습 문서
│   ├── kuka_communication.md    # KUKA 통신 + KRL(진공 포함) 설계 학습 문서
│   └── research_transparent_case_binpicking.md  # 투명 케이스 비전 조사 보고서
├── data/                        # (gitignore) 사용자 캡처/세션 데이터
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 안전 ⚠️

산업용 로봇은 **사람을 다치게 할 수 있는 기계**입니다. 본 프로그램은 다음과 같은 소프트웨어 안전장치를 갖추고 있지만, **하드웨어 안전 (안전 펜스, E-stop 회로, 안전 등급의 PLC)** 을 대체하지 않습니다.

소프트웨어 안전 기능:
- **AUT 모드 자동 50 % 속도 상한**
- **Z 좌표 하한 검증** (바닥 충돌 방지) — 단일 모션 + 시퀀스 큐 모두 적용, Approach 지점 Z 도 검증
- **Space 키 = 비상정지** (`robo_scram=TRUE` + KRL `BRAKE F` + `RESUME` 으로 현재 모션 취소)
- **이동 직전 확인 다이얼로그** (목표 좌표 + 속도 + 모드 미리보기)
- **회전 변화량 표시** (큰 IK 회전 예상 시 PTP 권장 경고)

권장 절차:
1. 첫 시도는 항상 **T1 모드, 5–10 % 속도**로 진행
2. 모션 큐에 명령 추가 후 SmartPAD에서 데드맨+시작 버튼으로 진행
3. AUT 모드는 안전 회로 점검 후에만 사용

---

## 알려진 한계

- **맞닿거나 겹친 객체**: 빈 픽킹 탭의 DBSCAN 3D ROI로는 분리 곤란 → CAD 매칭은 **전체장면 PPF**로 대응, 비-CAD는 **SAM3 분할**로 완화(완전 해결은 아님)
- **투명 객체 깊이**: 투명체는 깊이가 NaN → ROI 3D 필터·CAD 깊이 매칭이 약함. Grasp 설정의 "고정 평면 투영"으로 우회
- **속도**: OpenCV PPF 전체장면은 HALCON급 sub-1초까진 어려움(구현 최적화 격차) → 장면 다운샘플·샘플스텝으로 단축
- **윗면이 좌우 대칭인 객체**의 180° 모호성: 자동 결정 불가 → 수동 180° flip 토글 제공 (CAD 매칭 탭)
- **충돌 회피 없음**: Approach → Target 직선 경로에 다른 객체 있으면 충돌. ROI 박스·그리퍼 충돌 검사 미구현
- **`compute_approach_pose` 의 Tool +Z 가정**: KUKA `TOOL_DATA` 가 그리퍼 끝과 정렬되어야 함

---

## 학습 문서

이 프로젝트가 어떻게 동작하는지 학습 목적으로 정리한 한국어 문서들입니다.

- **[docs/hand_eye_calibration.md](docs/hand_eye_calibration.md)** — Hand-eye calibration 이란 / Eye-in-Hand vs Eye-to-Hand / 데이터 수집 / 5종 알고리즘 / 비선형 정밀화 / 3D 카메라가 mm 절대 측정이 가능한 이유
- **[docs/bin_picking.md](docs/bin_picking.md)** — RF-DETR/SAM3 검출 / ROI 필터 / OBB·여는 방향(이음선·밝기·격자) / 3D 포즈 추정 / 좌표 변환 / 수직 접근 / Grasp 설정(4-DOF) / 진공 픽 사이클 / 안전장치
- **[docs/cad_matching.md](docs/cad_matching.md)** — CAD 기반 6D pose / 매칭 알고리즘 4종(FPFH·PPF·전체장면 PPF) / DBSCAN·평면 제거 / Grasp 3D 설정 / 멀티 인스턴스·속도 튜닝
- **[docs/surface_tracking.md](docs/surface_tracking.md)** — 검은 선 검출(threshold·thinning) / skeleton path tracing / 3D 거리 샘플링 / 법선 자세 + offset / KRL 큐 동적 채움
- **[docs/kuka_communication.md](docs/kuka_communication.md)** — C3Bridge 프로토콜 / 메시지 포맷 / KRL 20슬롯 모션 큐 / 진공 인터럽트 계약 / Python ↔ KRL 흐름 / 비상정지 + RESUME / RobotControlMixin
- **[docs/research_transparent_case_binpicking.md](docs/research_transparent_case_binpicking.md)** — 투명 화장품 케이스 빈피킹 비전 기술 조사 보고서

---

## 기여

학습 프로젝트라 모든 기여를 환영합니다. 특히:
- 다른 KUKA 모델 / Zivid 모델에서의 동작 보고
- 알고리즘 개선 (특히 PPF 안정성, 6D pose 정확도)
- 영문 README / 문서 번역
- 충돌 회피, segmentation 기반 객체 분리 등 한계 항목 개선

---

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참고.

요약: 누구나 자유롭게 사용/수정/배포/상업 활용 가능. 단 원본 copyright 와 라이선스 notice 를 사본/파생물에 포함시켜야 함. 보증 없음.

---

## 감사

- **KUKA** — C3Bridge / KukaVarProxy 호환 프로토콜
- **Zivid** — 산업용 3D 카메라 + 친절한 SDK 문서
- **Open3D / OpenCV** — FPFH / ICP / PPF 표준 구현
- 본 시스템의 핵심 로직 다수가 [Drost et al. 2010 (PPF)](https://doi.org/10.1109/CVPR.2010.5540108), [Zhou et al. 2016 (FGR)](https://doi.org/10.1007/978-3-319-46475-6_47), [Park & Martin 1994 (Hand-eye)](https://doi.org/10.1109/70.326576) 등의 고전 논문에 기반함.
