# CAD 기반 6D Pose Matching

이 문서는 본 프로그램의 **CAD 매칭 탭**(`cad_matching_tab.py`)이 어떻게 객체의 6D 자세를 추정하고 로봇이 그것을 잡으러 가게 만드는지 학습 목적으로 정리한다. 사용법이 아니라 알고리즘의 원리와 본 시스템의 설계 결정에 초점을 둔다.

전제: hand-eye calibration이 끝나 `T_cam2base` (또는 `T_cam2gripper`)가 있고, Zivid 3D 카메라가 mm 단위 절대 좌표를 제공한다. 기초는 [hand_eye_calibration.md](hand_eye_calibration.md) 참고.

---

## 1. Bin Picking 탭과 무엇이 다른가

| | Bin Picking 탭 | CAD 매칭 탭 |
|---|---|---|
| 객체 인식 입력 | 학습된 **YOLO (vidnn)** | CAD 모델 (STL/OBJ/PLY) |
| 객체별 결과 | 중심점 + 표면 법선 (**5DoF**) | 완전한 6D 자세 (회전 3 + 위치 3) |
| 학습 데이터 필요? | YES | **NO** (CAD만 있으면 됨) |
| 무작위 자세 처리 | 약 (수직 접근 가정) | **강** (특히 PPF) |
| 객체 회전 방향 제어 | 못 함 | **가능** (CAD 좌표축 기준) |

CAD 매칭의 강점: **객체의 자세를 회전 포함 6개 자유도 모두 추정**한다. 그래서 길쭉한 객체를 그리퍼 손가락 방향과 정렬하거나, 객체가 비스듬히 누워있어도 정확히 잡을 수 있다.

---

## 2. 매칭 알고리즘 3종 — 시나리오별 강약

| 알고리즘 | 비결정적? | 무작위 자세 | 부분 가시성 | 본 시스템 사용 시점 |
|---|:-:|:-:|:-:|---|
| **FPFH + ICP (RANSAC)** | O | 약 (cull 필요) | 약 | 정자세 환경, 빠른 결과 필요 |
| **FPFH + ICP (FGR)** | X | 약 (cull 필요) | 약 | RANSAC 변동성을 피하고 싶을 때 |
| **PPF (OpenCV)** | O | **강** | **강** | 진짜 빈 픽킹 (무작위 자세, 가림) |

### 2.1 FPFH + ICP

- **FPFH (Fast Point Feature Histogram)**: 각 점 + 주변 점들의 법선 분포를 33D 히스토그램으로 인코딩. 점 단위 지역 기술자.
- **RANSAC**: scene과 model의 FPFH descriptor가 비슷한 점쌍 4개를 무작위로 골라 자세 가설을 만들고, 가장 inlier 많은 가설을 선택. 비결정적 — 매번 다른 결과.
- **FGR (Fast Global Registration)**: RANSAC 대체. Truncated least squares + tuple constraint로 결정적 매칭. 빠르고 같은 입력 → 같은 결과.
- **ICP refinement**: 위 글로벌 매칭의 초기 자세에서 Point-to-Plane ICP로 정밀화. 본 시스템은 **두 단계 ICP** (관대한 거리 → 좁힌 거리)로 평면 슬라이드 약점 보완.

한계: FPFH가 객체 표면 곡률에 민감해서 **부분만 보이는 무작위 자세**에 약함 → cull (visible-side only) 같은 우회가 필요한데 그건 정자세 가정.

### 2.2 PPF (Point Pair Features)

Drost et al. 2010 "Model Globally, Match Locally" 알고리즘. 본 시스템은 OpenCV `cv2.ppf_match_3d` 모듈 (contrib) 사용.

**핵심 아이디어**: 객체 표면의 **모든 점쌍 (p1, p2)** 에 대해 4D 특징을 계산해 해시 테이블에 저장. 매칭 시 scene 점쌍들의 4D 특징으로 voting → 가장 표 많이 받은 자세 채택.

```
4D 특징 F:
  F = ( ||p1-p2||,            ← 거리
        ∠(n1, p2-p1),         ← n1과 두 점 잇는 선의 각도
        ∠(n2, p2-p1),         ← n2와 두 점 잇는 선의 각도
        ∠(n1, n2) )           ← 두 법선 사이 각도
```

**왜 빈 픽킹에 강한가**:
- 무작위 자세 자동 처리 (모든 자세를 voting)
- 부분 가시성 강건 (객체 일부만 보여도 충분한 점쌍이 매칭되면 답 나옴)
- 학습 불필요 (CAD만 있으면 됨)

**비용**: 학습 단계에서 점쌍 N² 처리. 본 시스템은 학습 시 자동 다운샘플(`max_points_for_train=2500`)로 시간 폭주 방지.

자세한 PPF 원리: 본 시스템에 통합된 [`train_ppf_detector`](../cad_matching_tab.py), [`ppf_match_per_cluster`](../cad_matching_tab.py) 참고.

---

## 3. 전처리: 작업대 평면 제거 + DBSCAN 클러스터링

매칭 전 scene을 정제한다.

### 3.1 작업대 평면 제거

ROI 안의 scene 포인트클라우드에서 **RANSAC plane fitting** 으로 가장 큰 평면 (작업대 표면) 검출 → 해당 inlier 점들 제거. [`remove_table_plane`](../cad_matching_tab.py) 가 Open3D `segment_plane` 으로 한 번에 처리.

이게 없으면 작업대 점들이 한 거대 클러스터를 만들어 그 안에 객체가 묻힌다.

### 3.2 DBSCAN 클러스터링

남은 점들에 **DBSCAN** (밀도 기반 클러스터링) 적용 → 객체 단위로 분리:

```python
labels = scene_pcd.cluster_dbscan(eps=15.0, min_points=100)
# eps: 같은 클러스터로 묶일 점 사이 최대 거리(mm)
# min_points: 클러스터로 인정될 최소 점 수
```

분리된 각 클러스터에서 **개별적으로 매칭** → 클러스터당 한 객체 가정으로 RANSAC/PPF 가 두 객체 사이를 매칭하는 사고 방지. 또 멀티 인스턴스 처리가 자연스러워짐 (각 클러스터 = 한 인스턴스 후보).

**한계**: 객체끼리 맞닿거나 겹친 경우 한 클러스터로 묶임 → DBSCAN으론 분리 불가. SAM (Segment Anything) 같은 학습 기반 segmentation 이 필요한 시점.

---

## 4. 매칭 파이프라인 흐름

```
캡처 → ROI → 작업대 평면 제거 → DBSCAN 클러스터
   ↓ (각 클러스터마다)
   ↓
Algorithm 선택:
   ─ FPFH+ICP (RANSAC) :
        scene/model downsample → FPFH → RANSAC N번 (best 선택) → 두 단계 ICP
   ─ FPFH+ICP (FGR) :
        scene/model downsample → FPFH → FGR (결정적) → 두 단계 ICP
   ─ PPF :
        (학습 1회만) CAD → visible side 정렬 → trainModel
        scene_data 변환 → detector.match (voting)
        후보 상위 N개 → Open3D Point-to-Plane ICP 정밀화
   ↓
fitness/RMSE 평가 → 최선 후보 1개 선택 (클러스터당)
   ↓
T_object_cam (4×4 카메라 좌표계 객체 자세)
   ↓ (인스턴스 선택 후)
   ↓
Hand-eye T_calib 적용 → T_object_base
   ↓
Grasp 위치/회전 + Tool 정렬 → TCP 자세 (KUKA ABC)
```

---

## 5. Grasp 위치/회전 설정 — CAD 매칭 탭만의 기능

bin picking 탭은 객체 중심을 잡지만, CAD 매칭 탭은 **CAD 좌표계의 임의 위치를 잡는 점**으로 지정할 수 있다.

### 5.1 Grasp 위치 (X/Y/Z, CAD 좌표계 mm)

UI 의 spin 으로 입력. CAD 원점이 객체 외부(예: 모델링 좌표계 원점)에 있어도 사용자가 객체 중심이나 손잡이 위치로 grasp 점을 조정 가능. "객체 중심" 버튼 한 번이면 CAD bbox 중심으로 자동 설정.

매칭 결과 적용 시:
```python
grasp_in_base = T_object_base @ [gx, gy, gz, 1]
```
객체가 어떤 자세로 잡혀있든 grasp 점이 객체와 함께 회전 + 이동.

### 5.2 Grasp 회전 (A/B/C, Tool deg)

KUKA ZYX intrinsic Euler. **잡기 축 정렬 후 Tool 좌표계 기준** 추가 회전:

- **A** (Tool +Z 둘레, yaw): 그리퍼 손목 회전 — 평면 객체에서 손가락 정렬
- **B** (Tool +Y 둘레, pitch): 비스듬한 접근
- **C** (Tool +X 둘레, roll): 옆으로 기울임

(0, 0, 0) 이면 보정 없음. 잡기 축 + 뒤집기로 결정된 기본 자세에서 출발해 사용자가 미세 조정.

### 5.3 잡기 축 (`grasp_axis`)

콤보로 `Z / X / Y / Off (자동)` 선택. CAD 좌표계의 어느 축을 Tool +Z 에 정렬할지 결정.

- **Z 기본**: CAD +Z 가 객체 윗면이면 OK
- **Y / X**: CAD 가 다른 좌표축 규약으로 만들어진 경우
- **Off**: 객체 자세를 그대로 TCP 자세로 사용 (보정 없음). 무작위 자세 + 학습 기반에 적합

PPF 모드로 전환 시 잡기 축은 **유지**하지만 cull 옵션만 자동 OFF (PPF 가 무작위 자세에 강건하므로 cull 불필요). 다시 FPFH+ICP 로 돌아오면 cull 선호가 복원됨. 자세한 자동 토글 로직은 [`_on_algo_changed`](../cad_matching_tab.py).

### 5.4 6D pose → TCP 자세 변환

[`object_pose_to_tcp`](../cad_matching_tab.py) 가 모든 보정을 한 번에 처리:

```python
def object_pose_to_tcp(T_object_cam, T_calib, calib_mode, current_tcp,
                       grasp_axis, grasp_flip, grasp_offset_xyz,
                       grasp_rotation_abc_deg):
    # 1. 카메라 → 베이스 변환 (eye_to_hand / eye_in_hand)
    T_object_base = T_calib @ T_object_cam     # (eye_to_hand)
    # 또는 T_object_base = T_g2b @ T_calib @ T_object_cam  (eye_in_hand)

    # 2. Grasp 오프셋 적용 (객체 좌표계 기준)
    T_grasp_base = T_object_base @ translation(grasp_offset_xyz)

    # 3. Tool +Z 를 객체 grasp_axis 에 정렬 (flip 옵션 적용)
    R_tcp = align_tool_z_to_axis(T_grasp_base, grasp_axis, grasp_flip)

    # 4. Tool 좌표계 기준 ABC 회전 보정
    R_corrected = R_tcp @ Rotation.from_euler("ZYX", abc_rad).as_matrix()

    return homogeneous_to_tcp(T_grasp_base[:3, 3], R_corrected)
```

`grasp_axis="Off"` 면 객체 자세를 그대로 TCP 로 사용 + ABC 보정만 적용.

---

## 6. 클러스터당 후보 1개 출력 — 시각/시퀀스 일관성

PPF 는 voting 결과로 상위 N 개 후보를 내는데, 거의 동일한 자세가 여러 개 나와 3D 뷰에 겹쳐 보이는 문제가 있다. [`ppf_match_per_cluster`](../cad_matching_tab.py) 의 `n_show_per_cluster=1` 로 **클러스터당 best 1개만** 인스턴스로 출력 — fitness 기준 정렬 후 최상위 채택.

진단 로그(다이얼로그)엔 상위 5 개의 votes 가 모두 표시되어 voting 다양성/품질 확인 가능:
```
[클러스터 1] PPF 후보 51개, 상위 5개 votes=[1209, 1061, 1059, 953, 926]
  후보 1: votes=1209, ICP fitness=0.612, RMSE=1.234mm → 인스턴스 #1 채택
```

---

## 7. 시각화 — 사용자가 매칭/잡기 자세를 직접 확인

CAD 매칭 탭은 4개의 독립 뷰를 갖는다 (`view_stack`):

| 뷰 | 내용 |
|---|---|
| **2D 뷰** | Zivid 컬러 이미지 + ROI 드래그 + bbox 라벨 |
| **3D 뷰** | scene 포인트클라우드 + 매칭된 인스턴스(메시) + grasp 마커 + **Tool 자세 좌표축** |
| **CAD 뷰** | CAD 모델 + 좌표축 + cull 결과 + voxel 다운샘플 + grasp 위치 마커 |
| **클러스터 뷰** | DBSCAN 클러스터를 색깔별로 분리 표시 (eps/min_pts 조정용) |

### 인스턴스 메시 오버레이

매칭된 객체마다 CAD 메시를 인스턴스 색상으로 반투명 표시. 사용자가 매칭 정확도를 시각적으로 즉시 평가 가능.

### Tool 자세 시각화 (선택된 인스턴스)

bin picking 탭과 동일하게:
- **빨강/초록/파랑 축** = Tool X/Y/Z (50 mm)
- **주황 구** = Approach 지점 (Tool -Z 방향)
- **주황 선** = Approach → grasp 점 경로

좌표계 변환 주의: `target_pose` 는 베이스 좌표계인데 3D 뷰는 카메라 좌표계라 회전을 변환해야 한다 — `R_in_cam = T_calib[:3,:3].T @ R_target_base` 등. 자세한 변환은 [`_render_tcp_visualization`](../cad_matching_tab.py).

이 시각화 덕분에 grasp 위치/회전 spin을 조정하면 **실시간으로 그리퍼 자세가 어떻게 바뀌는지** 3D 뷰에 즉시 반영된다.

---

## 8. 180° flip 토글 — 좌우 대칭 객체의 모호성 보정

차단기처럼 윗면이 거의 좌우 대칭인 객체는 PPF/ICP 가 정자세 vs 180° 회전된 자세에 거의 동등하게 voting → 한쪽이 임의로 선택됨. 알고리즘적 자동 해결은 어려운 본질적 한계.

UI 의 **"180° 회전"** 버튼:
- `self._flip_applied` 플래그를 토글
- `_select_instance` 가 이 플래그를 보고 매번 Tool +Z 둘레 180° 회전을 적용 (영속)
- grasp 스핀 조정/재선택 후에도 사용자가 한 번 누른 flip 은 유지됨

새 인스턴스 선택 시(다른 객체) flip 자동 리셋 — 각 객체별로 독립.

---

## 9. 안전 + 시퀀스 큐 + Mixin

CAD 매칭 탭의 로봇 제어 / 시퀀스 큐 / 비상정지 / Home 이동 등은 빈 픽킹 탭과 완전히 동일하게 [`RobotControlMixin`](../robot_control_mixin.py) 을 상속해서 처리. 자세한 내용은 [bin_picking.md § 8, § 10.5](bin_picking.md) 와 [kuka_communication.md § 6–7](kuka_communication.md).

차이점:
- 시퀀스 라벨 명사: `SEQ_OBJECT_NOUN = "인스턴스"` (bin 은 "객체")
- 클러스터/매칭 결과 정리: 새 캡처/새 매칭 시 인스턴스 actor + grasp 마커 + TCP 시각화 actor 추적 리스트가 모두 비워짐 (stale 방지)

---

## 10. 한계와 다음 단계

- **객체끼리 맞닿/겹친 경우**: DBSCAN 으로 분리 불가 → SAM 같은 학습 기반 segmentation 필요
- **윗면 좌우 대칭**: PPF/ICP 자동 해결 불가 → 수동 180° flip 토글로 대응
- **PPF 학습 시간**: CAD 가 매우 크면 점쌍 N² 폭증 → `max_points_for_train`/`relative_sampling_step` 조정으로 trade-off
- **충돌 회피 없음**: Approach 직선 경로에 다른 객체 있으면 충돌 — 점유 격자 기반 모션 플래너가 다음 단계
- **6D pose 자체의 정확도 한계**: ICP RMSE ~1 mm 수준은 Zivid 노이즈에 가까운 한계. 더 정밀한 결과는 학습 기반 6D pose (FoundationPose, FFB6D)가 필요할 수 있음

---

## 11. 참고 자료

- Drost et al. (2010). *Model Globally, Match Locally: Efficient and Robust 3D Object Recognition.* CVPR — PPF 원논문
- Zhou et al. (2016). *Fast Global Registration.* ECCV — FGR
- Rusu et al. (2009). *Fast Point Feature Histograms (FPFH) for 3D Registration.* ICRA
- Open3D 문서: [Global Registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
- OpenCV Surface Matching: [cv2.ppf_match_3d module](https://docs.opencv.org/4.x/d9/d25/group__surface__matching.html)
- 본 시스템의 hand-eye calibration 원리: [hand_eye_calibration.md](hand_eye_calibration.md)
- 본 시스템의 KUKA 통신 구조: [kuka_communication.md](kuka_communication.md)
- 본 시스템의 빈 픽킹 (YOLO 기반): [bin_picking.md](bin_picking.md)
