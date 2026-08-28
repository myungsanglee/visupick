# CAD 기반 6D Pose Matching

이 문서는 본 프로그램의 **CAD 매칭 탭**(UI: `cad_matching_tab.py`, 알고리즘: `cad_registration.py` — 순수 모듈로 분리되어 Qt 없이 단독 실행/튜닝 가능)이 어떻게 객체의 6D 자세를 추정하고 로봇이 그것을 잡으러 가게 만드는지 학습 목적으로 정리한다. 사용법이 아니라 알고리즘의 원리와 본 시스템의 설계 결정에 초점을 둔다.

전제: hand-eye calibration이 끝나 `T_cam2base` (또는 `T_cam2gripper`)가 있고, Zivid 3D 카메라가 mm 단위 절대 좌표를 제공한다. 기초는 [hand_eye_calibration.md](hand_eye_calibration.md) 참고.

---

## 1. Bin Picking 탭과 무엇이 다른가

| | Bin Picking 탭 | CAD 매칭 탭 |
|---|---|---|
| 객체 인식 입력 | 학습된 **RF-DETR** | CAD 모델 (STL/OBJ/PLY) |
| 객체별 결과 | 중심점 + 표면 법선 (**5DoF**) | 완전한 6D 자세 (회전 3 + 위치 3) |
| 학습 데이터 필요? | YES | **NO** (CAD만 있으면 됨) |
| 무작위 자세 처리 | 약 (수직 접근 가정) | **강** (특히 PPF) |
| 객체 회전 방향 제어 | 못 함 | **가능** (CAD 좌표축 기준) |

CAD 매칭의 강점: **객체의 자세를 회전 포함 6개 자유도 모두 추정**한다. 그래서 길쭉한 객체를 그리퍼 손가락 방향과 정렬하거나, 객체가 비스듬히 누워있어도 정확히 잡을 수 있다.

---

## 2. 기초 개념 — 매칭을 이해하기 위한 최소 지식

이 장은 3D 매칭을 처음 접하는 사람을 위한 것이다. 3장의 알고리즘 설명은 여기 나오는 용어를 전제한다. (좌표계/변환 행렬 `T_a2b` 읽는 법은 [hand_eye_calibration.md § 1](hand_eye_calibration.md) 참고.)

### 2.1 점군(Point Cloud)과 "정합(Registration)"이라는 문제

- **점군**: 3D 점 `(x, y, z)` 들의 집합. 본 시스템에는 두 종류가 등장한다:
  - **Model**: CAD 파일 표면에서 샘플링한 점들 (기준 자세, mm 단위)
  - **Scene**: Zivid 카메라가 찍은 실제 장면의 점들 (카메라 좌표계, mm 단위)
- **정합(registration)**: "Model 점군을 어떻게 회전/이동시키면 Scene 속 실제 객체와 겹쳐지는가?"를 푸는 문제. 답은 4×4 변환 행렬 **T** 하나다.
- T 는 회전 3 자유도 + 이동 3 자유도 = **6DoF (Degrees of Freedom)**. 그래서 이 작업을 "6D pose estimation"이라 부른다. 크기(scale)는 풀지 않는다 — CAD 와 Zivid 둘 다 mm 절대 단위라 크기가 이미 맞기 때문 ([hand_eye_calibration.md § 6.5](hand_eye_calibration.md) 참고).

### 2.2 왜 두 단계(global → local)로 나누나

정합 알고리즘은 성격이 다른 두 부류가 있고, **반드시 둘을 이어서 쓴다**:

| | Global registration | Local refinement (ICP) |
|---|---|---|
| 초기 자세 필요? | 불필요 | **필요** (대충이라도 맞아야 함) |
| 결과 정밀도 | 거칠다 (수 mm ~ cm) | 정밀 (sub-mm) |
| 본 시스템의 예 | FPFH+RANSAC, FGR, PPF | Point-to-Plane ICP |

Global 이 "어디쯤 어떤 자세인지"를 대충 찾아주고, ICP 가 그 근처에서 정밀하게 붙인다. 하나만으로는 안 된다 — ICP 는 초기 자세가 틀리면 엉뚱한 지역해(local optimum)에 수렴하고, global 만으로는 mm 정밀도가 안 나온다.

### 2.3 Voxel 다운샘플 — 모든 파라미터의 기준 단위

원본 점군은 수십만 점이라 그대로 쓰면 느리다. 공간을 `voxel_size`(mm) 크기의 정육면체 격자로 나누고 **칸마다 점 하나만** 남기는 것이 voxel 다운샘플이다.

중요한 건 `voxel_size` 가 단순 성능 옵션이 아니라 **이후 모든 거리 임계값의 기준 눈금**이라는 점: FPFH 탐색 반경 = voxel×5, RANSAC 대응 임계 = voxel×2, ICP 거리 = voxel×1.5→1.0 ... UI 에서 voxel 하나만 바꿔도 파이프라인 전체의 "자"가 같이 바뀐다. CAD 로드 시 대각선/40 을 권장값으로 자동 제안하는 이유이기도 하다 (객체 크기에 눈금을 맞춤).

### 2.4 Descriptor(기술자) — 점의 "지문"

두 점군을 겹치려면 "Model 의 이 점 = Scene 의 저 점" 같은 **대응(correspondence)** 후보가 필요하다. 그런데 점 하나의 좌표만으로는 대응을 알 수 없다 (두 점군의 좌표계가 다르니까 좌표 비교는 무의미).

해법: **점 주변의 국소 형상**을 숫자 벡터로 요약한 것이 descriptor 다. 좌표계와 무관하게 "이 점 주변이 어떻게 생겼나"만 담는다.

- 평평한 면 위의 점, 모서리의 점, 구멍 가장자리의 점은 서로 다른 지문을 갖는다.
- Model 과 Scene 에서 **지문이 비슷한 점끼리 = 같은 부위일 후보**.
- **FPFH**(Fast Point Feature Histogram)는 그런 지문의 하나로, 주변 점들과의 법선 각도 관계를 **33개 숫자의 히스토그램**으로 요약한다.

### 2.5 RANSAC — 가설을 던지고 검증하기

대응 후보에는 오답이 많이 섞인다 (지문이 우연히 비슷한 다른 부위). RANSAC(RANdom SAmple Consensus)은 오답이 섞인 데이터에서 답을 찾는 일반 전략이다:

1. 대응 후보에서 **무작위로 최소 개수**(점 3~4쌍)만 뽑는다.
2. 그것만으로 변환 T 가설을 계산한다.
3. 가설대로 Model 을 옮겨보고, Scene 에 가까워진 점의 수(**inlier**)를 센다.
4. 1~3 을 수만 번 반복 → **inlier 가 가장 많은 가설**을 채택.

무작위 표본에 의존하므로 **실행할 때마다 결과가 조금씩 다르다**(비결정적). FGR 은 같은 문제를 최적화로 풀어 같은 입력 → 같은 결과(결정적)라는 차이가 있다.

### 2.6 ICP — 반복해서 딱 붙이기

ICP(Iterative Closest Point)는 이름 그대로를 반복한다:

1. 현재 자세에서 Model 각 점의 **최근접 Scene 점**을 임시 대응으로 삼는다 (임계 거리 이내만).
2. 그 대응들을 가장 잘 맞추는 강체 변환을 닫힌 형태로 푼다 ([hand_eye_calibration.md § 4.1(d)](hand_eye_calibration.md)의 Kabsch/SVD 와 같은 수학).
3. 변환을 적용하고 1 로 돌아간다. 대응이 더 안 바뀌면 수렴.

"최근접 점 = 진짜 대응"이라는 가정이 성립하려면 초기 자세가 대충 맞아야 한다 — 2.2절에서 global 단계가 먼저 필요한 이유가 이것이다. 변형 두 가지:

- **Point-to-Point**: 점↔점 거리를 최소화
- **Point-to-Plane**: 점↔(상대 점의 접평면) 거리를 최소화 — 평평한 면 위에서 미끄러지듯 정렬되어 더 빠르고 정확하게 수렴. **본 시스템이 쓰는 방식.**

### 2.7 매칭 품질 지표 — fitness 와 RMSE

진단 로그에 항상 나오는 두 숫자의 정확한 의미:

- **fitness** (0~1): Model 점 중 임계 거리 안에서 Scene 대응을 찾은 **비율**. "얼마나 많이 겹쳤나". 객체 일부가 가려져 있으면 보이는 부분만 붙으므로 1.0 이 나올 수 없다 — 그래서 빈 픽킹에선 0.4~0.7 도 정상 범위다.
- **inlier RMSE** (mm): 그 대응들의 평균 거리 오차. "겹친 부분이 얼마나 정확한가". Zivid 노이즈 수준(≲1mm)에 도달하면 사실상 한계 정밀도다.

인스턴스 채택 여부는 fitness 임계값(`fitness_spin`)으로 거른다 — RMSE 가 작아도 fitness 가 낮으면 "일부만 우연히 겹친" 오탐일 수 있기 때문.

---

## 3. 매칭 알고리즘 4종 — 시나리오별 강약

| 알고리즘 | 사전 분할 | 무작위 자세 | 부분 가시성 | 본 시스템 사용 시점 |
|---|:-:|:-:|:-:|---|
| **FPFH + ICP (RANSAC)** | 반복 제거 or DBSCAN | 약 (cull 필요) | 약 | 정자세 환경, 빠른 결과 |
| **FPFH + ICP (FGR)** | 반복 제거 or DBSCAN | 약 (cull 필요) | 약 | RANSAC 변동성 피할 때 |
| **PPF per-cluster** | **DBSCAN 필요** | **강** | 중 | 객체가 잘 떨어져 있을 때 |
| **PPF 전체장면** | **불필요** | **강** | **강** | 쌓임·부분 가림 (상용 방식, 권장) |

### 3.1 FPFH + ICP

- **FPFH (Fast Point Feature Histogram)**: 각 점 + 주변 점들의 법선 분포를 33D 히스토그램으로 인코딩. 점 단위 지역 기술자.
- **RANSAC**: scene과 model의 FPFH descriptor가 비슷한 점쌍 4개를 무작위로 골라 자세 가설을 만들고, 가장 inlier 많은 가설을 선택. 비결정적 — 매번 다른 결과.
- **FGR (Fast Global Registration)**: RANSAC 대체. Truncated least squares + tuple constraint로 결정적 매칭. 빠르고 같은 입력 → 같은 결과.
- **ICP refinement**: 위 글로벌 매칭의 초기 자세에서 Point-to-Plane ICP로 정밀화. 본 시스템은 **두 단계 ICP** (관대한 거리 → 좁힌 거리)로 평면 슬라이드 약점 보완.

한계: FPFH가 객체 표면 곡률에 민감해서 **부분만 보이는 무작위 자세**에 약함 → cull (visible-side only) 같은 우회가 필요한데 그건 정자세 가정.

### 3.2 PPF (Point Pair Features)

Drost et al. 2010 "Model Globally, Match Locally" 알고리즘. 본 시스템은 OpenCV `cv2.ppf_match_3d` 모듈 (contrib) 사용.

**핵심 아이디어**: 객체 표면의 **모든 점쌍 (p1, p2)** 에 대해 4D 특징을 계산해 해시 테이블에 저장(학습 1회). 매칭 시 scene 점쌍들의 4D 특징으로 voting → 가장 표 많이 받은 자세 채택.

**voting 이 실제로 어떻게 자세가 되나**: scene 에서 기준점 하나를 잡고, 다른 점들과 쌍을 만들어 4D 특징을 계산 → 해시 테이블에서 **같은 특징을 가진 model 점쌍들**을 조회 → 각 조회 결과는 "scene 기준점이 model 의 이 점이고, 이만큼 회전돼 있다"는 가설 하나 → 그 가설 칸에 한 표. 모든 점쌍을 처리하고 나면 **표가 누적된 칸 = 여러 점쌍이 동의하는 자세**가 후보로 떠오른다. 진단 로그의 `votes=[1209, 1061, ...]` 가 이 득표수다.

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

자세한 PPF 원리: 본 시스템에 통합된 [`train_ppf_detector`](../cad_registration.py), [`ppf_match_per_cluster`](../cad_registration.py) 참고.

### 3.3 PPF 전체장면 (DBSCAN 없음) — 상용 방식, 권장

PPF voting 은 원래 **전체 장면에 한 번** 돌려 다중 인스턴스를 직접 찾도록 설계된 알고리즘이다(Drost). 그런데 §3.2 의 `ppf_match_per_cluster` 는 먼저 DBSCAN 으로 장면을 나눈 뒤 **클러스터마다** PPF 를 돌린다. 이 "선(先)분할"이 두 가지 약점을 만든다:

- DBSCAN eps/min_points **튜닝 의존**.
- 객체가 **쌓여 붙으면** 한 클러스터에 여러 개가 뭉쳐 매칭 실패, **부분 가림**이면 클러스터가 조각남.

상용 프로그램(Photoneo, Pickit, MVTec HALCON `find_surface_model` 등)은 분할 없이 **CAD 로드 → 전체 장면에서 곧바로 다중 인스턴스**를 찾는다 — 이게 표면 기반 PPF 의 본래 강점이다. 국소 점쌍 voting 이라 물체가 부분만 보여도 그 조각의 점쌍이 자세에 투표하기 때문.

본 시스템의 [`ppf_match_whole_scene`](../cad_registration.py)(UI 알고리즘 = "PPF 전체장면"):

1. (선택) 장면 **voxel 다운샘플**(`scene_voxel`, UI voxel 값) → normal 추정·정렬 → `detector.match()` **1회** → voting 후보 다수.
2. **pre-ICP NMS**: raw voting 자세로 **먼저** 중복 제거 → 서로 다른 인스턴스 후보만 ICP (ICP 횟수 급감 = 속도↑, 놓침 위험 낮음).
3. 남은 후보를 **Open3D point-to-plane ICP** 로 정밀화(fitness/RMSE) → fitness 임계 통과분만.
4. **post-ICP NMS**: ICP 로 중심이 이동했을 수 있어 fitness 높은 순으로 다시 중복 제거(중심 거리 < model 대각×0.5) → 최대 `max_instances` 개.

**속도 — 계측과 지렛대**: `ppf_match_whole_scene` 은 각 단계(준비/voting/ICP/NMS) 소요 시간을 debug_log(결과 다이얼로그)에 `⏱ 준비 N / voting N / ICP N / NMS N ms` 로 남긴다. 실측(합성 2박스, 6800점): **voting 이 지배적**(1153ms) 이고 다운샘플(→1467점)하면 voting 243ms 로 ~5배 단축. 즉 **속도 최대 지렛대는 "장면 점 수 줄이기"**:
- **voxel**(장면 다운샘플) — 가장 큼. 단 너무 크게 하면 디테일이 뭉개져 **인스턴스를 놓침**(속도↔정확도 트레이드오프).
- **`relativeSceneSampleStep`**(장면 기준점 샘플링) 키우기.
- **pre-ICP NMS** 로 ICP 횟수는 이미 (후보 60 → 서로 다른 인스턴스 수) 로 절감됨.

> 검증(2 박스 합성 장면, 다운샘플 없음)에서 두 인스턴스를 fitness 1.0 으로 정확히 찾음. 약한 오검출(fitness ~0.14)은 fitness 임계(기본 0.15)로 걸러진다. OpenCV PPF 구현 특성상 HALCON 급 sub-1초는 어렵고, 위 지렛대로 입력 규모를 줄여 시간을 단축한다. 평면·대칭 모호성은 Open3D ICP + fitness 검증으로 억제.

---

## 4. 전처리: 작업대 평면 제거 + DBSCAN 클러스터링

> DBSCAN 은 FPFH per-cluster / PPF per-cluster 경로에만 쓰인다. **PPF 전체장면(§3.3)과 FPFH 반복 제거(`cad_match_multi_instance`)는 DBSCAN 을 쓰지 않는다.** 작업대 평면 제거는 네 경로 모두 공통.
> UI 에서 알고리즘을 **"PPF 전체장면"** 으로 고르면 DBSCAN 체크박스가 **자동 해제 + 비활성**(회색)된다 — 안 쓰는 옵션임을 드러내기 위함. 매칭 완료 시 상태바에 총 소요 시간(`매칭 NNms`)이 함께 표시된다.

매칭 전 scene을 정제한다.

### 4.1 작업대 평면 제거

ROI 안의 scene 포인트클라우드에서 **RANSAC plane fitting** 으로 가장 큰 평면 (작업대 표면) 검출 → 해당 inlier 점들 제거. [`remove_table_plane`](../cad_registration.py) 가 Open3D `segment_plane` 으로 한 번에 처리.

이게 없으면 작업대 점들이 한 거대 클러스터를 만들어 그 안에 객체가 묻힌다.

### 4.2 DBSCAN 클러스터링

남은 점들에 **DBSCAN** (밀도 기반 클러스터링) 적용 → 객체 단위로 분리. 원리는 단순하다: "가까운 점(eps 이내)이 충분히 많으면 같은 덩어리" — 서로 연결된 점들을 타고 가며 덩어리를 키우고, 어디에도 안 붙는 점은 노이즈로 버린다. 클러스터 개수를 미리 정할 필요가 없어서 (k-means 와 달리) 몇 개가 놓여 있는지 모르는 빈 픽킹에 맞는다:

```python
labels = scene_pcd.cluster_dbscan(eps=15.0, min_points=100)
# eps: 같은 클러스터로 묶일 점 사이 최대 거리(mm)
# min_points: 클러스터로 인정될 최소 점 수
```

분리된 각 클러스터에서 **개별적으로 매칭** → 클러스터당 한 객체 가정으로 RANSAC/PPF 가 두 객체 사이를 매칭하는 사고 방지. 또 멀티 인스턴스 처리가 자연스러워짐 (각 클러스터 = 한 인스턴스 후보).

**한계**: 객체끼리 맞닿거나 겹친 경우 한 클러스터로 묶임 → DBSCAN으론 분리 불가. SAM (Segment Anything) 같은 학습 기반 segmentation 이 필요한 시점.

---

## 5. 매칭 파이프라인 흐름

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

## 6. Grasp 설정 — CAD 매칭 탭만의 기능

bin picking 탭은 객체 중심을 잡지만, CAD 매칭 탭은 **CAD 좌표계의 임의 위치를 임의 자세로 잡도록** 지정할 수 있다.

### 6.1 Grasp 3D 설정 창 (PickIt 스타일)

"🎯 Grasp 3D 설정" 버튼을 누르면 별도 창([`GraspPointDialog`](../cad_matching_tab.py))이 열리고, CAD 를 솔리드 메쉬로 보면서 잡을 지점/자세를 직접 지정한다:

- **표면 클릭** → 그리퍼 끝 마커(주황 구)가 그 지점에 스냅
- **구 드래그 / X·Y·Z 스핀박스** → 위치 미세조정 ("원점", "객체 중심" 버튼 포함)
- **A·B·C 스핀박스** → Tool 좌표계 기준 회전 보정 (6.3절)
- **삼각대 화살표** (노랑 = Tool+Z 접근 방향, 빨강 = +X, 초록 = +Y) 가 **현재 A/B/C 값을 실시간 반영** — 적용 시 실제 로봇 자세와 어긋나지 않도록 같은 수학을 공유한다
- **"표면 법선으로 회전 자동"** 체크 시, 마커를 옮길 때마다 그 면에 수직으로 접근하도록 A/B/C 를 자동 계산 ([`suggest_rotation_from_normal`](../cad_registration.py) — 잡기 축 정렬 자세와 "Tool+Z = -법선" 자세의 차이 회전을 KUKA ZYX 로 분해)

### 6.2 왜 grasp 값을 CAD 좌표계로 저장하나

설정된 위치/회전은 **CAD 좌표계 기준**으로 저장된다 (`grasp_position_cad`, `grasp_rotation_abc_deg`). 매칭 결과 적용 시:

```python
grasp_in_base = T_object_base @ [gx, gy, gz, 1]
```

객체가 어떤 자세로 놓여 있든 **grasp 점이 객체와 함께 회전 + 이동**한다. 장면 좌표로 저장했다면 객체가 조금만 움직여도 무효가 되지만, CAD 좌표라서 "이 객체의 이 부위"라는 의미가 영구히 유지된다 — 한 번 설정하면 끝나는 이유.

### 6.3 Grasp 회전 (A/B/C, Tool deg)

KUKA ZYX intrinsic Euler. **잡기 축 정렬 후 Tool 좌표계 기준** 추가 회전:

- **A** (Tool +Z 둘레, yaw): 그리퍼 손목 회전 — 평면 객체에서 손가락 정렬
- **B** (Tool +Y 둘레, pitch): 비스듬한 접근
- **C** (Tool +X 둘레, roll): 옆으로 기울임

(0, 0, 0) 이면 보정 없음. 잡기 축 + 뒤집기로 결정된 기본 자세에서 출발해 사용자가 미세 조정.

### 6.4 잡기 축 (`grasp_axis`)

콤보로 `Z / X / Y / Off (자동)` 선택. CAD 좌표계의 어느 축을 Tool +Z 에 정렬할지 결정.

- **Z 기본**: CAD +Z 가 객체 윗면이면 OK
- **Y / X**: CAD 가 다른 좌표축 규약으로 만들어진 경우
- **Off**: 객체 자세를 그대로 TCP 자세로 사용 (보정 없음). 무작위 자세 + 학습 기반에 적합

PPF 모드로 전환 시 잡기 축은 **유지**하지만 cull 옵션만 자동 OFF (PPF 가 무작위 자세에 강건하므로 cull 불필요). 다시 FPFH+ICP 로 돌아오면 cull 선호가 복원됨. 자세한 자동 토글 로직은 [`_on_algo_changed`](../cad_matching_tab.py).

### 6.5 6D pose → TCP 자세 변환

[`object_pose_to_tcp`](../cad_registration.py) 가 모든 보정을 한 번에 처리:

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

## 7. 클러스터당 후보 1개 출력 — 시각/시퀀스 일관성

PPF 는 voting 결과로 상위 N 개 후보를 내는데, 거의 동일한 자세가 여러 개 나와 3D 뷰에 겹쳐 보이는 문제가 있다. [`ppf_match_per_cluster`](../cad_registration.py) 의 `n_show_per_cluster=1` 로 **클러스터당 best 1개만** 인스턴스로 출력 — fitness 기준 정렬 후 최상위 채택.

진단 로그(다이얼로그)엔 상위 5 개의 votes 가 모두 표시되어 voting 다양성/품질 확인 가능:
```
[클러스터 1] PPF 후보 51개, 상위 5개 votes=[1209, 1061, 1059, 953, 926]
  후보 1: votes=1209, ICP fitness=0.612, RMSE=1.234mm → 인스턴스 #1 채택
```

---

## 8. 시각화 — 사용자가 매칭/잡기 자세를 직접 확인

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

grasp 조정은 6.1절의 3D 설정 창에서 하고, 적용하면 CAD 뷰의 마커·매칭 결과 뷰의 인스턴스별 grasp 마커·선택 인스턴스의 TCP 시각화가 모두 갱신된다.

---

## 9. 180° flip 토글 — 좌우 대칭 객체의 모호성 보정

차단기처럼 윗면이 거의 좌우 대칭인 객체는 PPF/ICP 가 정자세 vs 180° 회전된 자세에 거의 동등하게 voting → 한쪽이 임의로 선택됨. 알고리즘적 자동 해결은 어려운 본질적 한계.

UI 의 **"180° 회전"** 버튼:
- `self._flip_applied` 플래그를 토글
- `_select_instance` 가 이 플래그를 보고 매번 Tool +Z 둘레 180° 회전을 적용 (영속)
- grasp 스핀 조정/재선택 후에도 사용자가 한 번 누른 flip 은 유지됨

새 인스턴스 선택 시(다른 객체) flip 자동 리셋 — 각 객체별로 독립.

---

## 9.5 매칭은 워커 스레드에서 돈다

"매칭 실행"의 무거운 계산(FPFH/PPF, 1~수십 초)은 UI 스레드가 아니라 **`MatchWorker(QThread)`** 에서 실행된다 ([cad_matching_tab.py](../cad_matching_tab.py)).

**왜 바꿨나:** 예전에는 계산이 UI 스레드에서 돌면서 `QApplication.processEvents()` 를 곳곳에 넣어 화면을 억지로 갱신했다. 이 방식은 (1) UI 반응성이 processEvents 를 얼마나 자주 부르느냐에 좌우되고, (2) **비상정지(Space)도 그 사이에만 동작**하는 안전 문제가 있었다. 연속 픽처럼 로봇이 자율로 움직이는 기능이 생기면서 "긴 계산 중에도 항상 반응하는 UI"가 안전 요구사항이 됐다.

**설계 규칙** (스레드 안전의 핵심):
1. **위젯 값은 시작 전에 UI 스레드에서 전부 읽어** `params` dict 로 클로저에 캡처한다 — 워커는 Qt 위젯을 절대 만지지 않는다.
2. 진행/결과/오류는 **시그널**(`progressed`/`finished_ok`/`failed`, 큐 연결)로 UI 스레드에 돌아와 처리된다 — `_on_matching_progress` / `_on_matching_done` / `_on_matching_failed`.
3. 장면은 ROI 크롭이 만든 새 포인트클라우드라 UI 와 공유되지 않고, PPF 학습 캐시는 완료 시점에 **CAD 경로가 그대로일 때만** 반영한다 (매칭 중 새 CAD 를 로드했으면 폐기).
4. 빠른 전처리(ROI 크롭, 평면 제거)와 검증은 그대로 UI 스레드 — 사용자에게 즉시 경고를 띄워야 하므로.

캡처(카메라 SDK)와 SAM3/RF-DETR 추론은 아직 UI 스레드다 — SDK 스레드 제약과 CUDA 컨텍스트(스레드별 push/pop 필요)가 얽혀 있어 실기 검증 없이 옮기기 위험해서 남겨 뒀다.

## 10. 안전 + 시퀀스 큐 + Mixin

CAD 매칭 탭의 로봇 제어 / 시퀀스 큐 / 비상정지 / Home 이동 등은 빈 픽킹 탭과 완전히 동일하게 [`RobotControlMixin`](../robot_control_mixin.py) 을 상속해서 처리. 자세한 내용은 [bin_picking.md § 8, § 10.5](bin_picking.md) 와 [kuka_communication.md § 6–7](kuka_communication.md).

차이점:
- 시퀀스 라벨 명사: `SEQ_OBJECT_NOUN = "인스턴스"` (bin 은 "객체")
- 클러스터/매칭 결과 정리: 새 캡처/새 매칭 시 인스턴스 actor + grasp 마커 + TCP 시각화 actor 추적 리스트가 모두 비워짐 (stale 방지)

---

## 11. 한계와 다음 단계

- **객체끼리 맞닿/겹친 경우**: DBSCAN 으로 분리 불가 → SAM 같은 학습 기반 segmentation 필요
- **윗면 좌우 대칭**: PPF/ICP 자동 해결 불가 → 수동 180° flip 토글로 대응
- **PPF 학습 시간**: CAD 가 매우 크면 점쌍 N² 폭증 → `max_points_for_train`/`relative_sampling_step` 조정으로 trade-off
- **충돌 회피 없음**: Approach 직선 경로에 다른 객체 있으면 충돌 — 점유 격자 기반 모션 플래너가 다음 단계
- **6D pose 자체의 정확도 한계**: ICP RMSE ~1 mm 수준은 Zivid 노이즈에 가까운 한계. 더 정밀한 결과는 학습 기반 6D pose (FoundationPose, FFB6D)가 필요할 수 있음

---

## 12. 참고 자료

- Drost et al. (2010). *Model Globally, Match Locally: Efficient and Robust 3D Object Recognition.* CVPR — PPF 원논문
- Zhou et al. (2016). *Fast Global Registration.* ECCV — FGR
- Rusu et al. (2009). *Fast Point Feature Histograms (FPFH) for 3D Registration.* ICRA
- Open3D 문서: [Global Registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
- OpenCV Surface Matching: [cv2.ppf_match_3d module](https://docs.opencv.org/4.x/d9/d25/group__surface__matching.html)
- 본 시스템의 hand-eye calibration 원리: [hand_eye_calibration.md](hand_eye_calibration.md)
- 본 시스템의 KUKA 통신 구조: [kuka_communication.md](kuka_communication.md)
- 본 시스템의 빈 픽킹 (RF-DETR 기반): [bin_picking.md](bin_picking.md)
