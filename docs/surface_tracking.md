# Surface Tracking (표면 추적)

이 문서는 본 프로그램의 **표면 추적 탭**([surface_tracking_tab.py](../surface_tracking_tab.py))이 어떻게 동작하는지 학습 목적으로 정리한다. 사용법이 아니라 "곡면 위에 그린 선을 로봇이 표면에 수직인 자세로 따라가게 만드는" 원리에 초점을 둔다.

전제: hand-eye calibration이 끝나 `T_cam2base` 가 있고, 3D 카메라가 픽셀마다 XYZ(mm)와 법선을 준다. 좌표 변환·수직 접근 자세 계산은 [bin_picking.md](bin_picking.md) 5·6장과 원리가 같으므로 여기서는 표면 추적 고유의 부분(선 검출 → 경로 → 샘플링)에 집중한다.

---

## 1. 무엇을 푸는가

**시나리오:** 자동차 외관처럼 굴곡진 표면에 매직펜으로 **검은 선**을 그린다. 카메라로 그 선을 인식해, 로봇 툴이 **선을 따라 이동하되 매 지점에서 표면에 수직(법선 방향)으로 정렬**한 채 시작점→끝점까지 움직이는 모션을 만든다. (도장·실링·검사·연마처럼 "곡면을 일정 자세로 훑는" 작업의 뼈대다.)

빈 픽킹/CAD 매칭과 다른 점:

| | Bin Picking / CAD | Surface Tracking |
|---|---|---|
| 대상 | 낱개 물체의 한 점(파지점) | **연속 경로**(선 위 수십~수백 점) |
| 자세 | 한 자세 | 점마다 **표면 법선 따라 달라지는 자세** |
| 입력 | 검출/매칭 | 사용자가 그린 **선 + 시작/끝점 클릭** |

---

## 2. 파이프라인 개요

[surface_tracking_tab.py](../surface_tracking_tab.py) 모듈 docstring의 9단계:

1. **캡처** — RGB + XYZ + 법선 맵 (`_capture`, [surface_tracking_tab.py:882](../surface_tracking_tab.py#L882)).
2. **검은 선 검출** → 1픽셀 두께 skeleton (`detect_line`, [:241](../surface_tracking_tab.py#L241)).
3. 사용자가 2D 뷰에서 **시작점/끝점 두 번 클릭** (`_on_image_click`, [:1014](../surface_tracking_tab.py#L1014)).
4. skeleton 위에서 **BFS 최단 경로** (`trace_path_on_skeleton`, [:332](../surface_tracking_tab.py#L332)).
5. **누적 3D 거리 기준 mm 간격 샘플링** (`sample_path_by_3d_distance`, [:389](../surface_tracking_tab.py#L389)).
6. 각 샘플의 **법선(국소 평면 피팅) → Tool +Z = −법선 자세**.
7. **offset mm** 만큼 법선 바깥으로 띄움 (표면에서 떨어뜨림).
8. 카메라 좌표계 → 로봇 base 좌표계 변환.
9. **QTimer 폴링으로 KRL 20슬롯 큐를 동적으로 채움** (경로가 슬롯보다 길어도 실행).

핵심 계산은 `_compute_path`([:1127](../surface_tracking_tab.py#L1127))에 모여 있다.

---

## 3. 검은 선 검출 → skeleton (2단계)

`detect_line`([:241](../surface_tracking_tab.py#L241)):

1. **Adaptive Threshold** — 조명이 균일하지 않은 곡면에서 전역 임계는 실패하므로, 국소 영역 평균 대비 어두운 픽셀을 선으로 잡는다(`cv2.adaptiveThreshold`, `THRESH_BINARY_INV`). 블록 크기(`block_size`)·상수(`threshold_c`)로 민감도 조절.
2. **Morphology open→close** — 잡티(open)를 지우고 선의 끊김(close)을 메운다.
3. **연결 성분 면적 필터** — `min_area` 미만의 작은 덩어리는 노이즈로 버려 **가장 큰 선만** 남긴다.
4. **Thinning(Zhang-Suen)** — `cv2.ximgproc.thinning` 으로 선을 **1픽셀 두께 중심선(skeleton)** 으로 만든다.

> **왜 thinning 인가:** 다음 단계의 경로 추적은 "선을 따라 한 픽셀씩 이동"하는 그래프 탐색이다. 선이 두꺼우면 어느 픽셀이 "중심"인지 모호해 경로가 갈라진다. **1픽셀 뼈대**로 만들면 선이 깔끔한 경로 그래프가 된다. (thinning 실패 시 mask 를 그대로 쓰는 폴백 있음.)

선택적으로 **ROI**(2D 드래그)를 지정하면 그 영역 안에서만 검출한다.

---

## 4. 시작/끝점 클릭 + BFS 경로 추적

사용자가 2D 뷰에서 선의 시작·끝을 클릭한다([`ClickPointImageLabel`](../image_view.py) — 짧은 클릭=점 지정, 드래그=ROI). 클릭이 정확히 skeleton 위가 아니어도, `trace_path_on_skeleton` 이 **가장 가까운 skeleton 픽셀로 스냅**한다.

그 뒤 **BFS(너비 우선 탐색)** 으로 시작→끝 최단 경로를 찾는다([:332](../surface_tracking_tab.py#L332)):

- skeleton 픽셀을 **노드**, 8방향 인접 skeleton 픽셀을 **간선**으로 보는 그래프에서 BFS.
- BFS 는 간선 가중치가 같은 그래프에서 **최단 경로**를 보장 → 갈래가 있어도 시작~끝 사이 한 줄을 뽑는다.
- `parent` 로 역추적해 경로 `[(x,y), ...]` 복원.

> **왜 BFS 인가:** skeleton 이 완벽히 한 줄이 아니라 잔가지가 있을 수 있다. BFS 최단 경로는 시작~끝을 잇는 **간선 수 최소 경로**를 뽑아 잔가지를 자연히 무시한다.

---

## 5. 3D 거리 기준 샘플링 (2D가 아니라 3D)

경로 픽셀은 촘촘하다(픽셀마다 하나). 로봇 모션은 그렇게 많이 필요 없으니 **일정 간격으로 샘플**한다 — 단, **2D 픽셀 간격이 아니라 실제 3D 거리 간격**으로 (`sample_path_by_3d_distance`, [:389](../surface_tracking_tab.py#L389)):

1. 경로 픽셀들의 대응 3D 점(`current_xyz`)을 따라 **누적 거리**를 계산.
2. `sampling_mm` 간격이 될 때마다 그 픽셀 인덱스를 선택(첫·끝점 포함).

> **왜 3D 거리인가:** 곡면에서는 카메라에 가까운/기울어진 부분이 2D에서 짧아 보인다. 2D 픽셀 간격으로 뽑으면 **실제 표면에선 간격이 들쭉날쭉**해진다. 3D 누적 거리로 뽑아야 로봇이 **표면 위 균등 간격**으로 움직인다. (깊이 NaN 픽셀은 건너뛴다.)

---

## 6. 점마다 법선 정렬 자세 + offset

각 샘플에서(`_compute_path` [:1207](../surface_tracking_tab.py#L1207)):

1. **법선 추정** — 카메라 법선맵이 있으면 쓰고, 없으면 그 픽셀 주변 패치의 3D 점으로 국소 평면 피팅(`estimate_normal_at_pixel`, [calibration.py](../calibration.py)). 패치 반경은 `법선 패치 반경(px)` spin.
2. **offset** — `shifted = 표면점 + 법선단위벡터 × offset_mm`. 즉 표면에서 **법선 바깥으로 offset mm 띄운** 지점이 TCP 목표. `offset=0` 이면 표면 접촉, 양수면 떨어뜨림.
3. **자세** — `compute_approach_pose(shifted, 법선, prev_tcp)` 로 **Tool +Z = −법선**(툴이 표면을 향함) 자세를 만든다. 원리는 [bin_picking.md §6](bin_picking.md) 과 동일.
4. **ABC unwrap** — 다음 점의 자세를 계산할 때 `prev_tcp` 를 직전 자세로 넘겨, 오일러 표현이 **이전 자세에 가까운 쪽**으로 풀리게 한다 → 점 사이 손목이 갑자기 한 바퀴 도는 것을 방지.

그런 다음 카메라 좌표계 → base 좌표계로 변환(eye-to-hand: `T_calib`, eye-in-hand: 현재 TCP 포함). 결과가 `self.path_points`(base TCP 자세 리스트).

---

## 7. 실행 — 긴 경로를 20슬롯 큐에 동적으로

경로 점이 수십~수백 개라 KRL **20슬롯 큐**보다 길 수 있다. 그래서 한 번에 다 넣지 않고 **동적으로 채운다** — 이 탭은 픽 사이클 대신 **자체 QTimer 폴링**을 쓴다(`_start_path_motion` → `_try_send_next`, [surface_tracking_tab.py:1296](../surface_tracking_tab.py#L1296)):

- 실행 시작 시 `_send_timer`(QTimer, 300ms 간격)를 돌린다.
- 매 tick 마다 `robot.has_empty_slot()` 로 빈 슬롯이 있으면 **다음 경로점 1개를 `add_move_lin` 으로 큐에 추가**([kuka_robot.py](../kuka_robot.py)).
- 모든 점을 송신하면 큐가 빌 때까지 기다렸다 종료.

각 점은 직전 점에서 이어지는 **LIN(직선) 이동**이라 표면을 매끄럽게 훑는다. 안전(Z 하한 `_validate_z`·AUT 속도 상한 `_effective_speed`·Space 비상정지)은 세 탭 공통 `RobotControlMixin` 에서 온다 ([kuka_communication.md §7](kuka_communication.md), [bin_picking.md §10.5](bin_picking.md)).

> 표면 추적 탭은 진공/픽 사이클·시퀀스 큐 UI는 쓰지 않지만(파지가 아니므로), Mixin 호환을 위해 일부 위젯을 숨김 스텁으로 둔다.

---

## 8. 조정값

| 값 | UI | 의미 |
|---|---|---|
| **offset(mm)** | `offset_spin` | 표면에서 법선 바깥으로 띄울 거리. 0=접촉, 음수=파고듦(주의) |
| **샘플 간격(mm)** | `sampling_spin` | 3D 거리 기준 점 간격. 작을수록 촘촘·모션 많음 |
| **법선 패치 반경(px)** | `normal_patch_spin` | 각 점 법선 추정용 국소 평면 패치 크기. 크면 매끈·둔감, 작으면 예민·노이즈 |
| block_size / threshold_c | 검출 파라미터 | adaptive threshold 민감도(어두운 선 검출) |

---

## 9. 한계

- **선이 잘 안 보이면**(반사·저대비) 검출이 끊긴다 → 조명·adaptive threshold 파라미터 조정.
- **깊이 NaN 구간**(투명·반사면)에서는 그 점의 법선/좌표를 못 구해 경로에서 빠진다.
- **충돌 회피 없음** — 경로가 다른 물체를 지나가면 툴이 부딪힐 수 있다.
- 법선 노이즈가 크면 점 사이 자세가 떨려 손목이 흔들릴 수 있다 → 패치 반경↑ 또는 샘플 간격↑로 완화.

---

## 10. 관련 문서

- 좌표 변환·수직 접근 자세(Tool +Z=−법선)·ABC unwrap: [bin_picking.md](bin_picking.md) §5·§6
- 로봇 큐·안전·Mixin: [kuka_communication.md](kuka_communication.md), [bin_picking.md §10.5](bin_picking.md)
