# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드다.

## 이 프로젝트는

**VisuPick** — KUKA 로봇(KRC5 + KSS 8.7)과 Zivid / RealSense 3D 카메라로 3D 비전 기반 빈 픽킹을 수행하는 단일 PySide6 데스크탑 앱. 하나의 GUI가 전체 워크플로우를 담당한다: 데이터 수집 → hand-eye calibration → 검증 → 빈 픽킹 / CAD 6D 자세 매칭 / 표면 추적.

UI 문자열, 코드 주석, `docs/` 학습 문서는 모두 **한국어**다 — 새로 추가하는 사용자 노출 문자열과 주석도 한국어로 맞춘다.

## 명령어

```bash
source venv/bin/activate       # venv/ 는 gitignore 대상이지만 로컬에 존재
pip install -r requirements.txt
python main.py                 # 유일한 진입점 — GUI 실행
```

**테스트 스위트, 린터, 빌드 단계가 없다.** 검증은 GUI를 통한 수동 확인이다 (실제 로봇 + 카메라 필요, 또는 SDK 스텁). `pytest`/`make`/CI가 있다고 가정하지 말 것.

## 아키텍처 — 큰 그림

아래의 파일 간 관계를 이해하는 것이 생산성의 지름길이다:

**공유 상태는 한 곳에 있다.** [main.py](main.py)의 `VisuPickApp(QMainWindow)`이 유일한 `self.robot`(`KUKARobot`)과 `self.camera`(`BaseCamera`)를 소유한다. 모든 탭을 생성할 때 *자기 자신*을 `main_window`로 넘기므로, 탭은 `self.main.robot` / `self.main.camera`로 하드웨어에 접근한다 — 탭이 연결을 직접 소유하지 않는다. 연결/해제 버튼도 탭이 아니라 메인 윈도우에 있다.

**탭 5개, 믹스인 2개:**
- [main.py](main.py)에 `DataCollectionTab`과 `VerificationTab`이 정의됨 (둘 다 `ImageViewerMixin` 사용).
- `BinPickingTab`([bin_picking_tab.py](bin_picking_tab.py)), `CADMatchingTab`([cad_matching_tab.py](cad_matching_tab.py)), `SurfaceTrackingTab`([surface_tracking_tab.py](surface_tracking_tab.py))은 모두 [robot_control_mixin.py](robot_control_mixin.py)의 `RobotControlMixin`을 상속한다. 이 믹스인이 *공유* 로봇 모션 UI를 제공한다: 단일 이동, 시퀀스 큐, Z 안전 한계, AUT 속도 상한, `Space` 키 비상정지. **이 세 탭의 로봇 안전/모션 동작은 탭별이 아니라 믹스인 한 곳에서 수정한다.**

**카메라 추상화 (GUI를 안 건드리고 카메라 추가):** [base_camera.py](base_camera.py)의 `BaseCamera(ABC)`가 인터페이스이고, [camera_factory.py](camera_factory.py)의 `_REGISTRY`(이름 → 모듈 경로 → 클래스명)가 `create_camera()` 호출 시점에만 해당 SDK를 지연 임포트한다 (SDK가 없어도 앱 시작이 깨지지 않음). 카메라 추가 방법: `BaseCamera` 구현 → `_REGISTRY`에 등록 → GUI 콤보에 자동 노출. 구현체: [zivid_camera.py](zivid_camera.py), [realsense_camera.py](realsense_camera.py), [percipio_camera.py](percipio_camera.py).

**KUKA 통신은 2계층** ([kuka_robot.py](kuka_robot.py)):
- `C3BridgeClient` — **TCP 포트 7000** 위의 저수준 C3Bridge / KukaVarProxy 프로토콜 (`read_variable` / `write_variable` / `send_motion`).
- `KUKARobot` — 그 위의 고수준 API: `get_tcp_position()`, **20슬롯 모션 큐**(`add_move_ptp` / `add_move_lin` / `..._rel`은 슬롯 번호 반환, `move_ptp` / `move_lin`은 블로킹 편의 래퍼), `emergency_stop`, `safety_pause/resume`, `set_speed`, `clear_queue`, 그리고 **진공 그리퍼 제어**(`set_vacuum` / `vacuum_blow` / `vacuum_release` — C3Bridge가 `$OUT` 직접 쓰기를 거부하므로 `robo_vac_*` 변수 + 트리거로 KRL이 대신 적용).
- 로봇 쪽에서는 [krl/ext_move.src](krl/ext_move.src)(`+ ext_move.dat`)가 그 큐를 처리하는 KRL 프로그램으로 돈다. Python이 큐에 넣고, KRL이 실행한다. 큐 깊이, 모션 타입(PTP/LIN), 비상정지 `RESUME` 인터럽트, 진공 인터럽트(83)는 **`KUKARobot`과 `ext_move.src` 사이의 계약**이다 — 반드시 양쪽을 함께 수정한다.

**캘리브레이션은 순수/Qt 독립** ([calibration.py](calibration.py)) — 단독으로 분석해도 안전하다. 핵심 함수: `tcp_to_homogeneous` / `homogeneous_to_tcp`(KUKA ABC 오일러 ↔ 4×4), `compute_hand_eye`(OpenCV 5개 방법 실행 → 비선형 정밀화 → greedy outlier 제거, 가장 일관성 좋은 결과 선택), `estimate_pose_from_pointcloud`(Zivid 경로) + `solvePnP` fallback(RealSense 경로), `compute_approach_pose`(그리퍼가 **Tool +Z** 방향으로 접근한다고 가정), `save/load_calibration_result`.

**선택적 RF-DETR 검출기** (빈 픽킹 전용) — [bin_picking_tab.py](bin_picking_tab.py)가 버튼 클릭 시점에 `from detector import Detector`를 지연 임포트한다. 환경변수 `RFDETR_DETECTOR_DIR`(`sys.path`에 추가)과 `RFDETR_MODEL_PATH`(`.engine`/`.onnx`)로 설정. 없어도 앱은 정상 시작하고 다른 탭은 모두 동작한다 — "객체 검출"만 안내 에러와 함께 실패. 클래스 매핑은 그 파일의 `RFDETR_CLASSES`이고, seg 모델의 마스크는 있으면 자동 활용된다.

## 규칙 & 함정

- **캘리브레이션 방식은 카메라 종류를 따라간다**: Zivid → `pointcloud`, RealSense → `pnp`, `compare`는 자동 선택. 한 경로만 하드코딩하지 말 것.
- **좌표/단위:** 위치는 mm, 각도는 KUKA `A B C` 오일러 도(deg). 동차 변환은 4×4 numpy. 변환은 반드시 `calibration.py` 헬퍼로만 한다.
- **사용자 세션 데이터**는 `data/session_*/pose_NNN/`에 쌓인다 (gitignore 대상). `calibration_result.json` / `intrinsics.json`도 gitignore 대상 — 생성 산출물은 절대 커밋하지 않는다.
- 알고리즘 심층 문서(한국어)는 [docs/](docs/)에 있다: `hand_eye_calibration.md`, `bin_picking.md`, `cad_matching.md`, `kuka_communication.md`. **알고리즘을 수정하기 전에 해당 문서를 먼저 읽는다.**
- **문서 작성 규칙** (docs/ 학습 문서):
  - **이 분야를 처음 접하는 개발자** 기준으로 쓴다: 모든 도메인 용어(법선, 정합, PTP, descriptor…)는 첫 등장에서 정의하고, 코드가 "무엇을 하는지"만이 아니라 **왜 그렇게 작성했는지**를 설명하고, 핵심 수식은 결과만 제시하지 말고 유도하고, 개념마다 구체적인 `file:line` 코드 참조를 연결한다. 로봇/비전 배경이 전혀 없는 독자가 문서 + 코드만으로 흐름을 따라갈 수 있어야 한다.
  - **문서가 설명하는 코드를 수정하면, 같은 작업 안에서 그 문서도 갱신한다** — UI 흐름, 계약(KRL 변수, I/O 매핑), 알고리즘 파라미터, 함수명 등. **낡은 문서는 버그로 취급한다.**
