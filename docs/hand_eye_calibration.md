# Hand-Eye Calibration

이 문서는 본 프로그램의 **데이터 수집 탭**에서 수행하는 hand-eye calibration의 원리와, 실제로 그 계산이 코드 안에서 어떻게 이루어지는지를 설명한다. 프로그램 사용법이 아니라 "왜 이렇게 동작하는가"를 학습 목적으로 정리한 문서다.

---

## 1. Hand-Eye Calibration이란?

로봇과 카메라를 같이 쓰는 시스템에서, **카메라가 본 좌표를 로봇이 움직일 좌표로 바꾸려면** 둘 사이의 정확한 위치 관계를 알아야 한다. 이 관계를 구하는 작업이 hand-eye calibration이다.

> "Hand"는 로봇의 손(=플랜지/그리퍼)이고, "Eye"는 카메라다. 즉, 손과 눈의 상대 위치를 구하는 작업.

### 왜 필요한가

카메라는 픽셀 좌표나 카메라 좌표계의 3D 좌표(예: `(120mm, -40mm, 850mm)`)를 출력한다. 그런데 로봇은 자기 베이스(또는 플랜지) 좌표계를 기준으로 움직인다. 카메라가 "여기 물체가 있어!"라고 알려줘도, 그 위치가 로봇 입장에서 어디인지 모르면 로봇은 갈 수 없다.

Hand-eye calibration은 다음 변환 행렬 중 하나를 구한다:

- **Eye-to-Hand**: `T_cam2base` — 카메라 좌표계 → 로봇 베이스 좌표계
- **Eye-in-Hand**: `T_cam2gripper` — 카메라 좌표계 → 로봇 플랜지(엔드이펙터) 좌표계

이 4×4 동차 변환 행렬(homogeneous transformation matrix)이 곧 "정답"이다.

### 4×4 변환 행렬 복습

```
T = [ R  t ]    R: 3x3 회전 행렬
    [ 0  1 ]    t: 3x1 평행이동 벡터
```

이 행렬 하나로 회전과 이동을 동시에 표현한다. 두 좌표계 사이의 변환을 합성할 때 행렬 곱으로 계산할 수 있어 편리하다.

```
T_a2c = T_b2c @ T_a2b      # A→B→C를 한 번에
```

---

## 2. 두 가지 방식: Eye-in-Hand vs Eye-to-Hand

### 2.1 Eye-in-Hand (카메라가 로봇에 붙어있음)

```
   [Robot Base]
        │
        │ T_gripper2base  (로봇이 알려주는 값)
        ▼
   [Gripper/Flange]
        │
        │ T_cam2gripper   ★ 우리가 구하려는 값
        ▼
   [Camera]
        │
        │ T_target2cam    (카메라로 측정)
        ▼
  [Checkerboard]
```

- **카메라가 로봇 손에 장착**되어 있는 경우.
- 로봇이 움직일 때 카메라도 같이 움직인다.
- 캘리브레이션 중에는 **체커보드를 월드(작업대)에 고정**하고 로봇을 여러 자세로 움직이면서 사진을 찍는다.
- 모든 자세에서 체커보드의 베이스 좌표는 일정해야 하므로, 이 일관성을 이용해 `T_cam2gripper`를 푼다.

**언제 쓰나?** 로봇이 시야를 바꿔가며 봐야 하는 작업 (검사, 추적, 좁은 공간 검색 등).

### 2.2 Eye-to-Hand (카메라가 외부에 고정)

```
   [Robot Base]                [Camera]
        │                          │
        │ T_gripper2base          │ T_target2cam
        ▼                          ▼
   [Gripper] ──── T_target2gripper ──── [Checkerboard]
                  (체커보드가 그리퍼에 붙어있음)

   ★ 구하려는 값: T_cam2base
```

- **카메라가 작업대 위 같은 외부에 고정**되어 있는 경우.
- 카메라는 안 움직이고 로봇만 움직인다.
- 캘리브레이션 중에는 **체커보드를 로봇 그리퍼(플랜지)에 부착**하고 로봇을 여러 자세로 움직이면서 사진을 찍는다.
- 모든 자세에서 체커보드의 그리퍼 기준 위치는 일정해야 하므로, 이 일관성을 이용해 `T_cam2base`를 푼다.

**언제 쓰나?** 빈 픽킹(bin picking), 작업 영역이 고정된 픽 앤 플레이스, 컨베이어 등.

> **본 프로그램의 기본 모드는 Eye-to-Hand**다. 사용자가 [main.py](../main.py) UI에서 콤보박스로 변경할 수 있다.

### 2.3 수학적 표현 (AX = XB 문제)

두 방식 모두 결국 같은 형태의 행렬 방정식을 푼다:

```
A_i · X = X · B_i        (i = 1, 2, ..., N)
```

- `X`: 우리가 구하려는 hand-eye 변환 (`T_cam2gripper` 또는 `T_cam2base`).
- `A_i`: 로봇 자세 i와 j 사이의 상대 변환 (로봇이 알려준 값에서 계산).
- `B_i`: 카메라 자세 i와 j 사이의 상대 변환 (체커보드를 측정해서 계산).

이 식의 의미는 직관적이다. "로봇이 A만큼 움직였을 때 카메라가 본 변화 B는, hand-eye 변환 X로 연결되어야 한다." 이 방정식을 여러 쌍에 대해 풀면 X를 결정할 수 있다.

OpenCV의 `cv2.calibrateHandEye()` 함수가 이 문제를 푸는 알고리즘을 5가지나 제공한다:
- **TSAI** (Tsai-Lenz, 1989) — 가장 고전적
- **PARK** (Park-Martin, 1994) — 회전을 lie algebra로 푸는 방식
- **HORAUD** (Horaud-Dornaika, 1995) — 비선형 최적화
- **ANDREFF** (Andreff et al., 1999) — 동시에 R과 t 추정
- **DANIILIDIS** (Daniilidis, 1999) — dual quaternion 기반

본 프로그램은 **모두 시도해보고 가장 일관성이 좋은 결과를 자동 선택**한다 (4.4절 참고).

---

## 3. 데이터 수집 방식

본 프로그램의 데이터 수집 탭이 하는 일을 단계별로 보자.

### 3.1 한 포즈에서 수집되는 것

"캡처" 버튼을 누를 때마다 다음 데이터를 한 묶음으로 저장한다:

| 항목 | 설명 |
|---|---|
| `tcp.json` | 로봇이 보고한 현재 TCP 위치 `(x, y, z, A, B, C)` (mm, deg) |
| `image.png` | Zivid 카메라의 2D 컬러 이미지 (BGR) |
| `pointcloud.zdf` | Zivid 원본 3D 프레임 (재처리 가능하도록 보관) |
| `pointcloud_xyz.npy` | (H, W, 3) 모양의 3D 포인트 클라우드 (mm 단위 XYZ) |
| `intrinsics.json` | 세션당 한 번 저장하는 카메라 내부 파라미터 |

저장 디렉토리 구조:
```
data/session_YYYYMMDD_HHMMSS/
├── intrinsics.json
├── pose_001/
│   ├── tcp.json
│   ├── image.png
│   ├── pointcloud.zdf
│   └── pointcloud_xyz.npy
├── pose_002/ ...
```

해당 코드:
- 캡처 트리거: [main.py:326](../main.py#L326) `_capture()`
- 저장: [main.py:368](../main.py#L368) `_save_pose()`
- 2D 이미지 추출: [zivid_camera.py:102](../zivid_camera.py#L102) `frame_to_2d_image()`
- 포인트 클라우드 추출: [zivid_camera.py:119](../zivid_camera.py#L119) `frame_to_point_cloud()`

### 3.2 좋은 데이터를 수집하는 원칙

Hand-eye calibration의 정밀도는 알고리즘보다 **데이터 다양성**에 훨씬 크게 좌우된다.

- 자세를 충분히 다양하게: 같은 위치에서 회전만 살짝 바꾼 자세는 사실상 정보가 없다. **회전축이 다른 자세**를 섞어야 한다.
- 카메라 시야 안에 체커보드가 또렷이 보이게.
- **최소 10개**, 가능하면 15~20개 자세를 권장.
- 체커보드가 너무 멀거나 너무 비스듬하면 검출 정밀도가 떨어진다.

### 3.3 KUKA TCP를 4×4 행렬로 변환

KUKA는 TCP 자세를 `(X, Y, Z, A, B, C)`로 표시한다. 여기서 `A=Rz`, `B=Ry`, `C=Rx`이고 **Z-Y-X 순서의 내재 회전(intrinsic rotation)**이다.

[calibration.py:19](../calibration.py#L19) `tcp_to_homogeneous()`:

```python
def tcp_to_homogeneous(tcp: Dict[str, float]) -> np.ndarray:
    t = np.array([tcp["x"], tcp["y"], tcp["z"]])
    a_rad = np.radians(tcp["a"])
    b_rad = np.radians(tcp["b"])
    c_rad = np.radians(tcp["c"])
    rot = Rotation.from_euler("ZYX", [a_rad, b_rad, c_rad])
    R = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
```

핵심은 `Rotation.from_euler("ZYX", ...)` 한 줄. KUKA의 오일러 각 규칙을 정확히 맞춰야 이후 모든 계산이 옳아진다.

---

## 4. 계산 과정

이제 본격적으로 데이터를 갖고 hand-eye 변환을 푸는 과정을 보자. 구현은 [calibration.py:201](../calibration.py#L201) `compute_hand_eye()`에 있다.

### 4.1 각 포즈에서 체커보드 자세 추정 (`T_target2cam`)

각 `pose_XXX` 폴더에서 다음을 한다:

#### (a) 2D 코너 검출

OpenCV의 `findChessboardCorners`로 체커보드 내부 코너의 픽셀 좌표를 찾고, `cornerSubPix`로 서브픽셀 정밀도로 보정한다.

[zivid_camera.py:189](../zivid_camera.py#L189) `detect_checkerboard()`:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
found, corners = cv2.findChessboardCorners(gray, board_size, flags)
if found:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
```

여기서 `board_size`는 **내부 코너 수**다. 8×6 체커보드라면 가로로 8개, 세로로 6개 = 총 48개 코너.

#### (b) 코너 픽셀 → 3D 좌표

일반적으로는 `cv2.solvePnP`로 카메라 내부 파라미터와 2D 코너만 가지고 자세를 추정한다. 하지만 본 프로그램은 **Zivid 3D 카메라가 제공하는 포인트 클라우드를 직접 활용**한다. 이게 일반적인 구현과 다른 점이다.

각 2D 코너 픽셀 `(px, py)` 위치에서, 포인트 클라우드의 같은 픽셀이 가지는 3D 좌표를 가져온다. 다만 단일 픽셀은 노이즈가 크므로 **주변 7×7 패치의 평균**을 사용한다.

[calibration.py:70](../calibration.py#L70) `_sample_3d_at_pixel()`:

```python
def _sample_3d_at_pixel(xyz, px, py, patch=3):
    h, w = xyz.shape[:2]
    x0, x1 = max(0, px - patch), min(w, px + patch + 1)
    y0, y1 = max(0, py - patch), min(h, py + patch + 1)
    region = xyz[y0:y1, x0:x1].reshape(-1, 3)
    valid = region[~np.any(np.isnan(region), axis=1)]
    if len(valid) == 0:
        return None
    return valid.mean(axis=0)
```

#### (c) 평면 제약 활용

체커보드는 **평면**이다. 이 사실을 강제하면 노이즈가 더 줄어든다. 모든 코너의 3D 점에 SVD 평면 피팅을 한 뒤, 각 점을 그 평면 위로 수직 투영한다.

[calibration.py:90](../calibration.py#L90) `_fit_plane()` — SVD로 최적 평면 구하기:

```python
def _fit_plane(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]   # 가장 작은 특이값 방향 = 평면 법선
    return centroid, normal
```

> **왜 SVD의 마지막 행이 법선인가?** SVD는 점들을 가장 잘 설명하는 직교 축들을 분산이 큰 순으로 정렬한다. 점들이 평면 위에 있으면, 평면 안에서의 분산은 크고 평면에 수직인 방향의 분산은 거의 0이다. 그래서 가장 작은 특이값에 해당하는 방향이 법선이 된다.

#### (d) Rigid transform (체커보드 좌표계 → 카메라 좌표계)

이제 두 점 집합이 있다:
- `obj_points`: 체커보드 자체 좌표계에서의 코너 위치 (예: 25mm 간격으로 격자).
- `cam_points`: 카메라 좌표계에서 측정한 3D 좌표.

이 둘 사이의 강체 변환(rigid transform: 회전 + 이동, 크기 변화 없음)을 SVD 한 번으로 푼다. 이게 곧 `T_target2cam`이다.

[calibration.py:178](../calibration.py#L178):

```python
obj_centroid = obj_points.mean(axis=0)
cam_centroid = cam_points.mean(axis=0)
obj_centered = obj_points - obj_centroid
cam_centered = cam_points - cam_centroid

H = obj_centered.T @ cam_centered
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:    # 반사(reflection) 방지
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

t = cam_centroid - R @ obj_centroid
```

이게 **Kabsch 알고리즘**이다. 점 대응이 알려진 두 집합 사이의 최적 회전을 닫힌 형태로 푸는 고전 알고리즘이다.

> **3D 직접 사용 vs `solvePnP`?**  
> `solvePnP`는 2D 픽셀과 카메라 내부 파라미터(왜곡 포함)에 의존한다. 카메라 캘리브레이션이 약간만 어긋나도 자세가 흔들린다. 반면 Zivid 같은 3D 카메라는 각 픽셀의 정확한 3D 좌표를 알려주므로 `solvePnP`를 우회하고 **3D-3D 매칭**으로 갈 수 있다. 더 직접적이고 보통 더 정밀하다. 포인트 클라우드 파일이 없을 때만 `solvePnP` fallback으로 떨어진다 ([calibration.py:275](../calibration.py#L275)).

### 4.2 OpenCV `calibrateHandEye()` 호출

여기까지 하면 각 포즈마다 두 변환이 모인다:
- `T_gripper2base` (TCP에서 변환)
- `T_target2cam` (위에서 추정)

Eye-to-Hand 모드에서는 전자를 역행렬로 뒤집어서 `T_base2gripper` 형태로 넣는다 — OpenCV는 두 모드 모두 같은 함수 인터페이스를 쓰기 때문에, 입력만 바꿔서 같은 함수로 푼다.

[calibration.py:312](../calibration.py#L312):

```python
if mode == "eye_to_hand":
    R_in_list = [R_g.T for R_g in R_gripper2base_list]
    t_in_list = [-R_g.T @ t_g for R_g, t_g in zip(R_gripper2base_list, t_gripper2base_list)]
elif mode == "eye_in_hand":
    R_in_list = R_gripper2base_list
    t_in_list = t_gripper2base_list
```

### 4.3 5가지 알고리즘 모두 시도

[calibration.py:403](../calibration.py#L403) `_try_all_methods()`:

```python
methods = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
for name, method in methods.items():
    R_out, t_out = cv2.calibrateHandEye(R_in_list, t_in_list,
                                         R_target2cam_list, t_target2cam_list,
                                         method=method)
    # ... 일관성 평가
```

각 알고리즘은 같은 문제를 다른 방식으로 푸는데, 데이터의 특성에 따라 잘 맞는 알고리즘이 다르다. 그래서 **다 돌려보고 평가지표가 가장 좋은 것을 선택**한다.

### 4.4 결과의 일관성 평가

평가지표: "체커보드의 위치가 모든 포즈에서 같은 곳에 있어야 한다"는 사실을 이용한다.

- **Eye-to-Hand**: 체커보드는 그리퍼에 붙어 있으므로 **그리퍼 좌표계 기준 체커보드 위치는 모든 포즈에서 같아야 한다**.
- **Eye-in-Hand**: 체커보드는 월드에 고정이므로 **베이스 좌표계 기준 체커보드 위치가 모든 포즈에서 같아야 한다**.

각 포즈에서 체커보드 위치를 계산해보고, **그 값들의 분산**이 작을수록 좋은 결과다.

[calibration.py:623](../calibration.py#L623) `_evaluate_hand_eye()`:

```python
for R_in, t_in, R_tc, t_tc in zip(...):
    T_in, T_tc = ...   # 4x4로 묶음
    T_target_in_ref = T_in @ T_result @ T_tc
    target_in_base_list.append(T_target_in_ref[:3, 3])

target_array = np.array(target_in_base_list)
mean_pos = target_array.mean(axis=0)
deviations = np.linalg.norm(target_array - mean_pos, axis=1)
return float(deviations.mean()), deviations.tolist()
```

평균 편차(mm 단위)가 작을수록 결과가 일관적이다. 잘 캘리브레이션된 시스템에서 보통 **1mm 이하**가 나오면 양호하다.

### 4.5 비선형 정밀화 (Non-Linear Optimization)

OpenCV의 5개 알고리즘은 모두 닫힌 형태의 해법이지만, 노이즈가 있으면 진짜 최적해는 아니다. 그래서 가장 좋은 결과를 초기값으로 두고 **비선형 최적화로 한 번 더 다듬는다**.

목적 함수는 4.4의 일관성 지표 자체. "각 포즈에서 계산한 체커보드 위치의 분산"을 최소화한다.

[calibration.py:577](../calibration.py#L577) `_refine_hand_eye()`:

```python
def residuals(params):
    T_est = _unpack_T(params)
    T_chain = T_in_arr @ T_est @ T_tc_arr
    positions = T_chain[:, :3, 3]
    mean_pos = positions.mean(axis=0)
    return (positions - mean_pos).flatten()

result = least_squares(residuals, _pack_T(T_init), method="lm", max_nfev=500)
```

`scipy.optimize.least_squares`의 Levenberg-Marquardt 알고리즘으로 푼다. 회전은 6개의 자유 파라미터(rotation vector 3 + translation 3)로 압축해서 다룬다 ([`_pack_T`](../calibration.py#L563), [`_unpack_T`](../calibration.py#L569)).

### 4.6 Greedy outlier 제거

자세 중에 측정 오차가 유난히 큰 게 섞일 수 있다 (체커보드가 흔들렸거나, 거리가 너무 멀었거나, 픽셀 노이즈가 컸거나). 이런 자세를 그대로 두면 다른 좋은 자세까지 끌려간다.

[calibration.py:329-383](../calibration.py#L329-L383)의 루프:

1. 모든 포즈로 캘리브레이션한다.
2. 포즈별 잔차(체커보드 위치 편차)를 계산한다.
3. **가장 잔차가 큰 포즈 하나를 빼고** 다시 캘리브레이션한다.
4. 평균 잔차가 줄어들면 채택, 그렇지 않으면 종료.
5. 이를 반복하되 **최소 8개 또는 원본의 절반**까지만 줄인다.

```python
while True:
    # ... 현재 keep_idx로 캘리브레이션
    refined_metric, refined_per_pose = _evaluate_hand_eye(...)
    
    if refined_metric < best_metric:
        best_T = refined_T; best_metric = refined_metric
    else:
        break   # 더 안 좋아짐 → 종료
    
    # 가장 큰 잔차 포즈 제거
    max_idx = int(np.argmax(refined_per_pose))
    keep_idx.pop(max_idx)
```

이 단계를 통해 실측 오차가 4mm 수준이던 결과가 0.2mm 수준까지 떨어지는 것을 확인했다.

---

## 5. 전체 흐름 요약

```
[데이터 수집]                    [계산]
                                 │
사용자 → "캡처" 버튼 ──┐         ▼
                       │   각 pose_XXX 폴더에서:
Zivid → 2D 이미지      │    ① cv2.findChessboardCorners → 2D 코너
      → 3D 포인트클라우드│    ② 코너 픽셀의 3D 좌표 (패치 평균)
                       │    ③ 평면 피팅 + 투영
KUKA → TCP (X,Y,Z,A,B,C)│   ④ Kabsch (SVD) → T_target2cam
                       │    ⑤ tcp_to_homogeneous → T_gripper2base
                       │
저장: tcp.json,        │   모든 포즈 모이면:
      image.png,       └─→ ⑥ cv2.calibrateHandEye (5가지 다 시도)
      pointcloud.zdf,      ⑦ 일관성 평가 → 최선 선택
      pointcloud_xyz.npy   ⑧ 비선형 최적화로 정밀화
                           ⑨ Greedy outlier 제거 반복
                           ▼
                      T_cam2base (또는 T_cam2gripper)
                      → calibration_result.json 저장
```

---

## 6. 알아두면 좋은 개념

### 6.1 동차 좌표(Homogeneous coordinates)

3D 점 `(x, y, z)`를 4차원 `(x, y, z, 1)`로 표기하면, 회전과 이동을 **하나의 4×4 행렬 곱셈**으로 합칠 수 있다. Hand-eye 분야에서 모든 포즈를 4×4로 다루는 이유가 이것이다.

### 6.2 좌표계 변환 표기법

이 코드베이스에서는 변환을 `T_a2b` 형태로 표기한다 — "a 좌표계의 점을 b 좌표계로 옮기는 변환".

```python
p_in_b = T_a2b @ p_in_a
```

연쇄 합성:
```python
T_a2c = T_b2c @ T_a2b      # 행렬은 오른쪽 → 왼쪽으로 적용
```

### 6.3 회전 표현 종류

| 표현 | 자유도 | 장단점 |
|---|---|---|
| 오일러 각 (X,Y,Z 등) | 3 | 직관적, 짐벌락 위험 |
| 회전 행렬 (3×3) | 9 (제약 6) | 합성이 행렬곱, 보간 어려움 |
| 회전 벡터 (rotvec / axis-angle) | 3 | 최적화에 좋음 |
| 쿼터니언 | 4 (제약 1) | 보간/합성 모두 깔끔 |

본 코드에서는 KUKA 입력은 오일러, 내부 계산은 회전 행렬, 비선형 최적화 시에는 회전 벡터로 변환해 쓴다.

### 6.4 카메라 내부 파라미터 (Intrinsics)

- **camera_matrix**: `fx, fy` (초점거리), `cx, cy` (광축 중심).
- **dist_coeffs**: 렌즈 왜곡 계수 (`k1, k2, p1, p2, k3`).

본 프로그램에서는 3D 직접 매칭을 쓰기 때문에 이 값이 보조용이다 (`solvePnP` fallback에서만 본격적으로 쓰임). [zivid_camera.py:150](../zivid_camera.py#L150) `get_intrinsics()`에서 Zivid SDK가 제공한 값을 받아 저장한다.

### 6.5 왜 3D 카메라는 거리와 무관하게 mm 단위 절대 측정이 가능한가?

본 시스템의 매칭(체커보드 자세 추정, CAD-scene 매칭 모두)이 잘 동작하는 **근본적 이유**다. 이해해두면 왜 일반 RGB 카메라보다 3D 카메라가 빈 픽킹에 압도적으로 유리한지 명확해진다.

#### 2D 카메라의 한계: 원근 모호성 (Perspective Ambiguity)

```
       카메라
        ●
       /|\
      / | \      ← 같은 픽셀 크기로 보이는 두 물체
     /  |  \
   📦   📦📦
  가까이  멀리
  (작음) (큼)
```

일반 2D 카메라 픽셀은 **빛의 강도(밝기)만** 담는다. 결과:

- 멀리 있는 큰 물체와 가까이 있는 작은 물체가 **사진상 똑같은 크기**로 보인다.
- 절대 크기를 알 수 없음 → 매칭 시 6DoF pose 외에 **scale 1개 자유도가 추가**되어 7DoF 문제가 된다.
- 이 모호성을 풀려면 객체의 실제 크기를 미리 알거나(예: 체커보드 한 칸 25mm), 별도 깊이 정보가 필요.

#### Zivid 3D 카메라의 차이

각 픽셀이 **(R, G, B, X, Y, Z)** — 색상 + **mm 단위 절대 3D 좌표**를 함께 담는다:

- 가까이 있으면 → 픽셀의 Z 값이 작음 (예: 500mm)
- 멀리 있으면 → 픽셀의 Z 값이 큼 (예: 1500mm)
- **물체의 실제 크기는 두 경우 모두 동일하게 측정됨** (Z만 다를 뿐)

→ Scale은 항상 1.0. 매칭은 **6DoF만 풀면 됨**.

#### 어떻게 절대 mm를 측정하나? (Triangulation)

```
   카메라 ●     ● 프로젝터
          \   /
   알려진  \ / 알려진
   거리    X   각도
  (baseline) \
              \
               🎯 객체 표면
```

핵심 원리는 **삼각측량**이다. 사람이 두 눈으로 거리를 가늠하는 것과 같은 원리(stereoscopic vision):

1. 프로젝터가 **알려진 패턴**(보통 격자 줄무늬 또는 코드화된 빛)을 객체에 투사.
2. 객체 표면에 비친 패턴이 카메라의 **어느 픽셀**에 잡히는지 측정.
3. **카메라-프로젝터 거리(baseline)** + **카메라 시야각** + **픽셀 위치**가 사전에 calibrate되어 있으므로 → 삼각측량으로 그 표면점까지의 거리를 계산.

단순화한 수식:
```
Z = (baseline × focal_length) / disparity
```

모든 값이 mm/픽셀 단위로 calibrate되어 있어 결과도 mm 단위로 나온다. Zivid는 공장 출고 시 이 calibration이 이미 정교하게 되어 있고, SDK가 그 값을 사용해 `xyz` 데이터를 mm로 제공한다.

#### 본 시스템 매칭이 이걸 활용하는 방식

```
Model (CAD)              Scene (Zivid)
같은 mm 단위              같은 mm 단위
↓                         ↓
점쌍 거리 ||p1-p2||      점쌍 거리 ||q1-q2||
   ↓                         ↓
   ↓── 직접 비교 가능 ──↓
        (단위가 같으니까)
```

| 매칭 단계 | mm 단위가 어떻게 쓰이나 |
|---|---|
| FPFH descriptor | 점쌍 거리 (mm) + 각도가 입력 |
| PPF | 4D 특징 중 거리 (mm)를 양자화해서 해시 키 생성 |
| RANSAC distance threshold | `voxel * 1.5` = mm 단위 임계값 |
| ICP correspondence | mm 단위 거리로 inlier 판단 |
| RMSE | mm 단위 정밀도 척도 |
| Hand-eye calibration | 모든 잔차/일관성 metric이 mm 단위 |

진단 로그에서 본 값들이 다 mm 단위인 이유가 이것이다:
```
ICP RMSE = 1.081mm    ← 실제 거리, 의미 있는 단위
model diag = 194.8mm  ← CAD 실제 크기
일관성 오차 mean = 0.196mm  ← 캘리브레이션 품질
```

매칭이 찾는 변환 행렬 **T**는 **회전(3) + 평행이동(3) = 6DoF**:

```
T = [ R  t ]    R: 회전 행렬 (3x3, 단위 무차원)
    [ 0  1 ]    t: 평행이동 (mm 단위 3-vector)
```

**T에 scale 항이 없음** = 변환이 객체 크기를 바꾸지 못함 = CAD 그대로의 크기로 scene과 매칭됨. 이게 "딱 맞게 인식되는" 이유다.

#### 거리가 변하면 어떻게 동작하는가?

예를 들어 차단기를 카메라에서 500mm 거리에 두든 1000mm에 두든:

- **2D 픽셀 크기는 절반으로 줄어들지만**...
- **Zivid가 측정한 3D XYZ 좌표는 정확한 mm 위치 그대로** (Z만 다름).
- Model 차단기 196mm × scene 차단기 196mm = **1:1 매칭**.
- 매칭된 변환 T의 평행이동 부분(`t`)만 거리에 따라 달라짐.
- 회전 `R`과 fitness는 **거리에 영향 받지 않음**.

#### 만약 2D 카메라였다면?

본 시스템에서 Zivid를 일반 RGB 카메라로 바꾼다면 대안:

| 방식 | 동작 |
|---|---|
| **`solvePnP`** | 카메라 intrinsics + 2D 코너 + model 3D를 알면 6DoF 추정 가능. 노이즈에 민감. 본 코드의 [calibration.py:275](../calibration.py#L275) fallback이 이것 |
| **딥러닝 6DoF pose estimation** | PVN3D, FoundationPose, FFB6D 등. GPU + 학습 데이터 필요 |
| **스테레오 카메라** | 카메라 2개로 직접 깊이 계산 (Zivid 원리와 동일) |

3D 카메라 가격이 떨어지면서 산업용에서 점점 표준이 되고 있다. 트레이드오프: 가격 vs 정밀도 vs 시야각 vs 깊이 정확도.

#### 정리 — 본 시스템 정밀도의 근원

차단기 매칭이 1mm 수준 RMSE로 나오는 건 다음이 다 맞물려서 가능:

1. **Zivid가 mm 단위 절대 3D 측정** → scale 자유도 제거
2. **CAD도 mm 단위** → 1:1 직접 비교 가능
3. **Hand-eye calibration이 0.2mm 일관성** → 카메라↔로봇 매핑이 정확
4. **모든 알고리즘(FPFH/PPF/ICP)이 절대 단위 활용** → 임계값이 의미 있음

2D RGB 카메라만 있는 시나리오에서는 같은 정밀도 달성이 훨씬 어렵다. 본 시스템의 강력함의 절반 이상은 Zivid 같은 산업용 3D 카메라 덕분이다.

### 6.6 포즈 추정 방식 선택 (`auto` / `pointcloud` / `pnp` / `compare`)

지금까지 설명한 4.1 ~ 4.6 의 흐름은 **3D 직접 매칭(Kabsch SVD)** 이 메인이고 `solvePnP` 가 fallback. 그런데 어느 게 정답인지는 **카메라에 따라 다르다**:

| 카메라 깊이 정밀도 | RGB intrinsics 품질 | 권장 방식 |
|---|---|---|
| 높음 (Zivid) | 보통 | **`pointcloud`** (3D 직접 매칭) |
| 낮음 (RealSense D415) | 양호 (공장 calibration) | **`pnp`** (solvePnP) |
| 잘 모름 | 잘 모름 | **`compare`** (둘 다 시도 후 일관성 비교) |

[`compute_hand_eye`](../calibration.py) 에 `pose_method` 인자가 있어 강제 선택 가능:

```python
T = compute_hand_eye(data_dir, ..., pose_method="pnp")     # solvePnP 강제 (RealSense 등)
T = compute_hand_eye(data_dir, ..., pose_method="pointcloud")  # 3D 직접 매칭 강제
T = compute_hand_eye(data_dir, ..., pose_method="auto")    # 기존 자동 (포인트클라우드 우선)
```

**비교 모드** 는 같은 데이터로 두 방법을 각각 풀고 일관성 metric 을 비교 (UI 데이터 수집 탭의 "포즈 추정: compare" 옵션):

```python
res_pc  = compute_hand_eye(..., pose_method="pointcloud", return_metric=True)
res_pnp = compute_hand_eye(..., pose_method="pnp",        return_metric=True)
# res_*["metric_mean"] = 각 포즈에서 체커보드 위치 분산 (mm)
# 더 작은 쪽이 더 일관성 좋음 → 그 결과 채택
```

결과 다이얼로그에 두 방법의 metric 이 같이 나오고, **차이의 크기로 신뢰도 진단**:
- 차이 < 0.5 mm: 두 방법 모두 신뢰 가능 (카메라 깊이/intrinsics 모두 양호)
- 차이 0.5 – 2 mm: 보통 (한쪽이 약간 더 정밀)
- 차이 > 2 mm: 한쪽이 부정확 — 카메라 종류와 맞춰 선택해야 함

이 비교 모드는 **새 카메라 환경에서 어느 방식이 본 시스템에 적합한지 객관적으로 결정**하는 도구. 한 번 비교한 결과로 평소 사용할 `pose_method` 를 결정하면 됨.

---

## 7. 참고 자료

- Tsai, R. Y. & Lenz, R. K. (1989). *A new technique for fully autonomous and efficient 3D robotics hand/eye calibration*. IEEE Trans. Robotics and Automation.
- Park, F. C. & Martin, B. J. (1994). *Robot sensor calibration: solving AX = XB on the Euclidean group*. IEEE Trans. Robotics and Automation.
- OpenCV docs: `cv2.calibrateHandEye` — https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Zivid SDK docs: 포인트 클라우드 / `experimental.calibration.intrinsics`
- Geng, J. (2011). *Structured-light 3D surface imaging: a tutorial*. Advances in Optics and Photonics — 3D 카메라 측정 원리
- Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* — 원근 모호성과 깊이 추정의 표준 교과서
