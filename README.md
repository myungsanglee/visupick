# VisuPick

> **3D vision-guided bin picking workbench for KUKA robots + Zivid 3D camera**
> Hand-eye calibration · 6D pose estimation · Bin picking · 통합 GUI

`VisuPick`은 산업용 로봇(KUKA KRC5 + KSS 8.7) 환경에서 Zivid 3D 카메라로 객체를 인식하고, hand-eye calibration을 거쳐 로봇이 정확히 picking 하도록 돕는 통합 데스크탑 워크벤치입니다. **데이터 수집 → 캘리브레이션 → 검증 → bin picking / CAD 매칭** 까지 4단계 워크플로우를 하나의 PySide6 GUI에서 처리합니다.

이 프로젝트는 학습 및 연구 용도로 작성되었고, 알고리즘이 어떻게 동작하는지 [docs/](docs/) 폴더에 상세 설명서가 포함되어 있습니다.

---

## 주요 기능

### 1. Hand-Eye Calibration
- **Eye-to-Hand** / **Eye-in-Hand** 두 모드 지원
- 체커보드 + Zivid 3D 포인트 클라우드 기반 정밀 추정 (`solvePnP` fallback 포함)
- OpenCV 5종 알고리즘 자동 비교 (TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS) → 가장 일관성 좋은 결과 선택
- 비선형 정밀화 + greedy outlier 제거 → **일관성 0.2 mm 이하 달성**
- 자세한 원리: [docs/hand_eye_calibration.md](docs/hand_eye_calibration.md)

### 2. Bin Picking (YOLO + 3D 포인트클라우드)
- `vidnn` YOLO 검출 → 객체별 3D 중심 + 표면 법선 추정
- 2D 이미지 ROI 드래그 + 3D 시점 자동 매칭
- 객체 중심에 **Tool 좌표축 시각화** (그리퍼 접근 자세를 실시간 미리보기)
- 시퀀스 큐: 여러 객체 픽업 순서 + Home 이동을 묶어 자동 실행
- 자세한 원리: [docs/bin_picking.md](docs/bin_picking.md)

### 3. CAD 기반 6D Pose Matching
- **FPFH + ICP (RANSAC / FGR)** — 정자세 환경에 빠르고 정확
- **PPF (OpenCV Surface Matching) + Open3D ICP 정밀화** — 무작위 자세 + 부분 가시성에 강건
- **DBSCAN 클러스터 분리** + **작업대 평면 자동 제거**로 빈 픽킹 시나리오 지원
- Grasp 위치(X/Y/Z) + 회전(ABC) 설정 → 사용자가 객체별 잡는 자세 정밀 조정
- 인스턴스 클릭 시 3D 뷰에 Tool 좌표축 + Approach 경로 시각화

### 4. 통신 / KRL
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
| 카메라 ① | **Zivid 2 M70** (구조광 3D, ~0.1 mm 정밀) |
| 카메라 ② | **Intel RealSense D415** (Active Stereo, ~2 mm 정밀) |
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

### vidnn (옵션, Bin Picking 탭 사용 시)

별도 YOLO 추론 라이브러리. 환경변수로 경로 설정:

```bash
# ~/.bashrc 또는 venv 활성화 스크립트에 추가
export VIDNN_PATH=/path/to/vidnn
export VIDNN_MODEL_PATH=/path/to/vidnn/runs/your_model.pt
```

설정 안 하면 default(`/home/robotegra/michael/vidnn/...`)가 사용되는데, 본인 환경엔 그 경로가 없을 테니 위 환경변수를 꼭 설정. vidnn 자체가 없거나 사용 안 하면 본 프로그램은 정상 시작하고 **Bin Picking 탭에서 "객체 검출" 버튼만 비활성**. CAD 매칭 탭 / 캘리브레이션은 vidnn 없이 동작.

---

## 실행

```bash
python main.py
```

GUI가 열리면 다음 순서로 진행:

1. **카메라 연결** + **카메라 설정 (YML)** 로드
2. **로봇 연결** (IP, Tool 번호 입력)
3. 작업 시나리오 선택:
   - **데이터 수집** 탭 → 캘리브레이션
   - **검증** 탭 → 캘리브레이션 정확도 확인
   - **Bin Picking** 탭 → YOLO + 픽킹
   - **CAD 매칭** 탭 → CAD 6D pose + 픽킹

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
├── bin_picking_tab.py           # Bin Picking 탭 (YOLO + 3D)
├── cad_matching_tab.py          # CAD 6D 매칭 탭 (FPFH/PPF/FGR)
├── robot_control_mixin.py       # 두 탭 공통: 로봇 이동·시퀀스 큐·안전·E-stop
├── calibration.py               # Hand-eye 알고리즘 + 객체 자세 계산
├── kuka_robot.py                # C3Bridge 통신 + 큐 기반 로봇 제어
├── base_camera.py               # 카메라 추상 인터페이스 (BaseCamera)
├── camera_factory.py            # 카메라 종류 → 인스턴스 생성 (지연 import)
├── zivid_camera.py              # Zivid SDK 래퍼
├── realsense_camera.py          # Intel RealSense SDK 래퍼 (D415 검증)
├── krl/
│   ├── ext_move.src             # KRL 모션 큐 프로그램 (KRC에 업로드)
│   └── ext_move.dat             # DEFDAT
├── config/                      # Zivid 카메라 설정 (YML, Zivid Studio에서 export)
│   ├── zivid_settings_manufacturing_specular.yml
│   └── zivid_settings_parcels_reflective.yml
├── cad_models/                  # 매칭에 사용할 CAD 파일 (STL/OBJ/PLY)
│   └── MCCB_Metasol_125af-3p.stl
├── docs/
│   ├── hand_eye_calibration.md  # 캘리브레이션 알고리즘 학습 문서
│   ├── bin_picking.md           # 빈 픽킹 파이프라인 학습 문서
│   ├── cad_matching.md          # CAD 기반 6D pose 매칭 학습 문서
│   └── kuka_communication.md    # KUKA 통신 + KRL 설계 학습 문서
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

- **객체끼리 맞닿거나 겹친 빈 픽킹**: DBSCAN으로 분리 불가 → SAM 같은 segmentation 또는 학습 기반 6D pose 필요
- **윗면이 좌우 대칭인 객체**의 180° 모호성: 자동 결정 불가 → 수동 180° flip 토글 제공 (CAD 매칭 탭)
- **충돌 회피 없음**: Approach → Target 직선 경로에 다른 객체 있으면 충돌
- **`compute_approach_pose` 의 Tool +Z 가정**: KUKA `TOOL_DATA` 가 그리퍼 끝과 정렬되어야 함

---

## 학습 문서

이 프로젝트가 어떻게 동작하는지 학습 목적으로 정리한 한국어 문서들입니다.

- **[docs/hand_eye_calibration.md](docs/hand_eye_calibration.md)** — Hand-eye calibration 이란 / Eye-in-Hand vs Eye-to-Hand / 데이터 수집 / 5종 알고리즘 / 비선형 정밀화 / 3D 카메라가 mm 절대 측정이 가능한 이유
- **[docs/bin_picking.md](docs/bin_picking.md)** — Bin picking 4단계 / YOLO 검출 / 3D 포즈 추정 / 좌표 변환 / 수직 접근 자세 계산 / Tool 자세 시각화 / 안전장치
- **[docs/cad_matching.md](docs/cad_matching.md)** — CAD 기반 6D pose / FPFH+ICP vs PPF / DBSCAN 클러스터링 / 작업대 평면 제거 / Grasp 위치·회전 / 멀티 인스턴스
- **[docs/kuka_communication.md](docs/kuka_communication.md)** — C3Bridge 프로토콜 / 메시지 포맷 (Type 0/1/11) / KRL 20슬롯 모션 큐 / Python ↔ KRL 흐름 / 비상정지 인터럽트 + RESUME / RobotControlMixin

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
