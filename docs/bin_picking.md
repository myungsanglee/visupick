# Bin Picking (빈 픽킹)

이 문서는 본 프로그램의 **빈 픽킹 탭**이 어떻게 동작하는지를 학습 목적으로 정리한다. 사용법이 아니라 "카메라가 본 물체 위치를 로봇이 어떻게 집을 자세로 변환하는가"의 원리에 초점을 둔다.

전제: hand-eye calibration이 이미 끝나서 `T_cam2base` 변환 행렬이 있는 상태. 캘리브레이션 자체는 [hand_eye_calibration.md](hand_eye_calibration.md) 참고.

---

## 1. Bin Picking이란?

빈 픽킹은 "박스(bin) 안에 무작위로 쌓여있는 물체를 로봇이 하나씩 집어내는 작업"이다. 단순한 컨베이어 픽 앤 플레이스와 다르게:

- 물체의 **위치**가 매번 다르다 → 카메라가 매번 봐야 한다.
- 물체의 **자세(회전)**도 매번 다르다 → 어디로 어떻게 다가갈지 매번 계산해야 한다.
- 옆 물체를 건드리지 않고 정확히 한 개만 집어야 한다 → **법선 방향(수직 접근)**이 중요하다.

> **용어 — 법선(normal)**: 표면에 **수직인 방향 벡터**. 물체 윗면의 법선은 하늘을 향하고, 경사면의 법선은 비스듬히 향한다. 흡착이든 집기든 "표면에 수직으로 접근"이 기본이므로, 이 문서 전체에서 법선 = 그리퍼가 다가갈 방향의 기준이다.

이걸 풀려면 다음 4단계가 필요하다:

1. **카메라로 박스 내부 촬영** → 2D 이미지 + 3D 포인트 클라우드 + 법선 맵.
2. **객체 검출** → 2D bbox로 물체 위치 후보 찾기.
3. **3D 정보 결합** → 각 bbox 안의 3D 점들로 중심과 법선 계산.
4. **로봇 자세 계산** → 카메라 좌표계 → 로봇 좌표계 변환 + 표면에 수직인 TCP 자세 만들기.

본 프로그램은 정확히 이 4단계를 거친다.

---

## 2. 데이터 입력 (Zivid 캡처)

[bin_picking_tab.py:917](../bin_picking_tab.py#L917) `_capture()`에서 다음을 가져온다:

| 변수 | 모양 | 의미 |
|---|---|---|
| `current_image` | (H, W, 3) BGR | 2D 컬러 이미지 (객체 검출용) |
| `current_xyz` | (H, W, 3) | 각 픽셀의 카메라 좌표계 3D 좌표 (mm) |
| `current_normals` | (H, W, 3) | 각 픽셀의 표면 법선 벡터 (Zivid SDK 제공) |
| `current_intrinsics` | (3, 3) | 카메라 내부 파라미터 (3D ↔ 2D 투영용) |

> **3D 카메라가 핵심이다.** 일반 RGB 카메라라면 깊이가 없어서 2D bbox만으로는 절대 위치를 알 수 없다. Zivid는 모든 픽셀에 대해 mm 단위의 정확한 XYZ를 알려준다 (구조광 방식 측정).

---

## 3. 객체 검출 (RF-DETR)

본 프로그램은 RF-DETR 추론 래퍼 `detector.py`(별도 리포, 환경변수로 경로 지정)를 사용한다 ([bin_picking_tab.py:1030](../bin_picking_tab.py#L1030) `_detect()`). 엔진 로드는 비싸므로 `Detector` 인스턴스는 1회만 생성해 캐싱하고, 신뢰도 임계값만 매 추론마다 갱신한다.

```python
from detector import Detector

self._detector = Detector(
    model_path=self.RFDETR_MODEL_PATH,     # .engine(TensorRT) 또는 .onnx
    model_name="rf-detr",
    class_names=self.RFDETR_CLASSES,       # {0: ladybug, 1: heart, 2: wings}
    conf_thresh=self.conf_spin.value(),    # 신뢰도 임계값
)
_, result = self._detector.predict(self.current_image)
```

출력 `result`는 병렬 배열을 담은 dict: `xyxy` (N, 4), `confidence` (N,), `class_id` (N,), `class_name`, 그리고 seg 모델이면 `mask` (N, H, W) bool (det 모델이면 `None`). 이를 downstream 호환을 위해 `[{"bbox", "confidence", "class_id", "class_name", ("mask")}, ...]` list-of-dict 로 변환한다. seg 모델의 mask 는 이후 크롭/표시에 자동으로 사용된다.

### 3.1 ROI 필터링

검출 결과에 박스(bin) 외부의 노이즈가 섞일 수 있다. 사용자가 2D 이미지에서 ROI(Region of Interest)를 드래그해 지정하면, 그 영역의 3D 좌표 범위 안에 있는 검출만 남긴다.

[bin_picking_tab.py:1083](../bin_picking_tab.py#L1083):

```python
if self.roi_3d is not None:
    filtered = []
    for det in detections:
        bcx = int((bbox[0] + bbox[2]) / 2)
        bcy = int((bbox[1] + bbox[3]) / 2)
        pt = self.current_xyz[bcy, bcx]  # bbox 중심의 3D 좌표
        if (roi_3d["x_min"] <= pt[0] <= roi_3d["x_max"]
            and roi_3d["y_min"] <= pt[1] <= roi_3d["y_max"]
            and roi_3d["z_min"] <= pt[2] <= roi_3d["z_max"]):
            filtered.append(det)
```

필터는 **2D 사각형(사용자가 그린 `roi_2d`)을 1차 게이트**로 쓰되, 검출의 **bbox 전체가 ROI 사각형 안에 완전히 포함될 때만** 통과시킨다(한쪽이라도 삐져나오면 제외). 그런 다음 검출 중심 픽셀의 **깊이가 유효하고 `roi_3d`가 있을 때만 깊이 밴드를 2차로** 확인한다. 이렇게 하면 불투명 객체는 "같은 바닥 vs 옆 선반"을 깊이로 분리하는 장점을 유지하면서(2차), **투명 객체(SAM3 대상)처럼 중심 깊이가 NaN 인 경우에도 2D 게이트로 ROI가 정상 적용**된다.

> 과거에는 `roi_3d`(깊이) 단독으로 필터링했는데, 투명 케이스는 (a) ROI 영역 깊이가 대부분 NaN 이라 `roi_3d` 자체가 설정 안 되고(유효점 10개 미만) (b) 검출 중심 깊이도 NaN 이라, ROI 필터가 통째로 무력화되는 문제가 있었다. 2D 1차 게이트로 해결.

### 3.2 대안 검출 경로 — SAM 3 텍스트 프롬프트 (검출기 불필요)

RF-DETR 은 학습된 클래스만 검출한다. 학습 없이 임의 객체를 찾으려면 탭 2행의 **"SAM3 텍스트" 입력 + "SAM3 검출" 버튼**을 쓴다 — Meta 공식 SAM 3(facebookresearch/sam3)의 개념 분할(Promptable Concept Segmentation)로, 명사구(예: `cosmetic case`) 하나만으로 **Grounding DINO 같은 별도 검출기 없이** 모든 인스턴스의 마스크를 만든다.

- 구현: [bin_picking_tab.py](../bin_picking_tab.py) `_detect_sam3()`. `sam3.model_builder.build_sam3_image_model` + `Sam3Processor` 로 로드(1회 캐싱), `set_image()` → `set_text_prompt()` → `(masks, boxes, scores)`.
- 선택적 의존성: SAM 3 미설치면 버튼만 안내 에러, 앱·다른 기능은 정상. repo 경로는 환경변수 `SAM3_MODEL_DIR`.
- 라이선스: SAM 3 는 Meta 커스텀 "SAM License"(상업 사용 가능, 군사/핵 등 금지 분야만 제외). SAM 1/2 는 Apache 2.0. Ultralytics 경유(AGPL) 대신 **공식 repo 직접 사용**.
- `Conf` 스핀 값이 점수 임계값으로 함께 적용됨. RF-DETR 과 SAM3 결과는 동일한 `detections` 포맷으로 변환되어 공통 다운스트림 `_apply_detections()`(ROI 필터 → 3D 포즈 → 2D/3D/테이블)를 탄다.

> SAM 3 는 무겁다(수억 파라미터, GPU 권장, 이미지당 초 단위). 실시간 사이클보다는 다품종·비정형 객체를 학습 없이 잡아야 할 때 유용.

### 3.3 마스크 → OBB(회전 사각형)

**OBB(Oriented Bounding Box)** = 축 정렬이 아니라 **객체를 가장 꽉 감싸는 회전된 사각형**이다. 일반 bbox(수평/수직)는 비스듬히 놓인 케이스를 헐겁게 감싸 방향 정보를 못 주지만, OBB 는 **중심 · 크기(가로·세로) · 회전각**을 함께 준다 — 로봇이 케이스의 긴 축에 맞춰 그리퍼를 정렬할 때 필요한 값이다.

탭 1행의 **"OBB 검출" 버튼**([bin_picking_tab.py](../bin_picking_tab.py) `_detect_obb`)은 검출된 각 객체의 **마스크**에 고전 CV 를 적용한다. bbox 만으로는 회전각을 알 수 없으므로 **반드시 마스크가 있는 검출**(seg 모델 또는 SAM3)에만 동작한다.

- 파이프라인 (`_obb_from_mask`): 마스크 → `cv2.findContours`(외곽 윤곽선) → 가장 큰 윤곽선 → `cv2.minAreaRect`. `minAreaRect` 는 그 점들을 감싸는 **최소 넓이 회전 사각형** `((cx,cy),(w,h),angle)` 을 반환한다.
- 결과는 `det["obb"] = {center, size, angle, box_pts(4개 꼭짓점)}` 로 저장하고, 2D 뷰에 회전 사각형 + 각도를 bbox 와 **같은 색**으로 그린다(색은 `_object_color_map()` 한 곳에서 관리).

> `minAreaRect` 의 `angle` 은 **180° 모호성**이 있다 — 사각형은 180° 돌려도 똑같이 겹치므로, 축(장/단축)만 알 뿐 "어느 쪽이 앞인가"는 모른다. 이 방향(앞/뒤)을 정하는 것이 다음 3.4 단계다.

### 3.4 여는 방향 추정 (클램셸 힌지/뚜껑)

**문제:** 화장품 케이스(클램셸)를 놓을 때 방향을 통일하려면 **여는 쪽(뚜껑이 열리는 방향)**을 알아야 한다. OBB 가 장축은 주지만 180° 모호성 때문에 "힌지가 두 변 중 어느 쪽인가"는 아직 모른다. 이 판별을 **고전 CV(이음선 에지 비대칭)** 로 한다 — VLM 불필요.

**여는 축은 단축으로 고정한다:** 화장품 클램셸은 **힌지와 립(여는 쪽)이 둘 다 긴 변**이다. 따라서 여는 방향은 두 긴 변 사이 = **짧은 extent 축(단축)** 을 향한다. 여는 축 자체는 OBB 의 단축으로 못박고, 이음선 비대칭은 **그 축의 부호(둘 중 어느 긴 변이 립인가)** 만 정한다. (여는 축까지 자동 선택하게 두면, 인쇄·힌지 금속 같은 잡음이 장축 오프셋을 키워 엉뚱하게 짧은 변 쪽으로 화살표가 나가는 문제가 있었다.)

**핵심 통찰 (왜 이음선인가):** 클램셸은 뚜껑과 바닥이 만나는 **이음선(seam, parting line)** 이 있는데, 이 선은 **여는 쪽 립 + 양 옆 2면, 즉 3면에 U자로** 나타나고 **힌지 쪽엔 없다**(그쪽은 뚜껑이 이어져 있으므로). 따라서 케이스 **내부의 에지 에너지(밝기 변화)가 여는 쪽으로 치우친다**. 투명 케이스는 인쇄가 적어 이 물리 이음선이 에지의 대부분을 차지하므로 특히 잘 맞는다.

**두 가지 방식(탭 3행 "여는 방향 방식" 콤보에서 선택):**

| 방식 (`method`) | 가중치 맵 `w(px)` | 원리 | 부호 의미 |
|---|---|---|---|
| **이음선 에지** (`"seam"`) | Sobel 에지 크기 `√(gx²+gy²)` | 뚜껑-바닥 이음선(여는 쪽+양옆 U자)의 에지가 여는 쪽에 몰림 | 에지 많은 쪽 = 립(여는 쪽) |
| **내부 밝기 비대칭** (`"brightness"`) | 밝기 `gray − min(gray)` | 투명 케이스 내부(팬/거울/힌지)의 밝기가 앞뒤로 비대칭 | 밝은 쪽 = 무게중심이 향하는 쪽(제품마다 다름 → 필요시 반전) |

두 방식은 **공통 뼈대**([bin_picking_tab.py](../bin_picking_tab.py) `_detect_opening` → `_opening_from_weight`)를 쓰고, **가중치 맵만 다르다**(`_opening_weight_map`):

1. 방식별 **가중치 맵**을 만든다(위 표).
2. 마스크를 **침식(erode)** 해 외곽 실루엣(케이스 vs 배경)을 제외 — 내부 특징만 남긴다(실루엣은 앞/뒤 대칭이라 단서가 안 됨).
3. 침식된 지지영역의 **기하 중심**과 **가중 무게중심**(=`Σ px·w / Σ w`)의 차 `off = 무게중심 − 기하중심` 을 구한다. 균일하면 0, 한쪽에 몰리면 그쪽으로 치우친다. (음수 가중치는 `clip(w,0,·)`으로 0 처리 → 무게중심 정의가 깨지지 않게.)
4. `off` 를 OBB **단축(짧은 extent 축)에 투영**한다. 여는 축은 단축으로 고정돼 있으므로, 투영의 **부호만 여는 방향**을 정한다. → 360° 단위벡터 하나로 180° 모호성이 풀린다.
5. **방향 반전** 체크 시 벡터를 180° 뒤집는다. 결과 `det["opening"] = {dir, angle_deg, confidence, half_len, axis, method}`. 2D 뷰에 **중심 → 여는 쪽 화살표**를 그린다.

**조정값 (탭 3행) — 하나씩 자세히:**

- **침식%** (`opening_erode_spin`, 범위 0–25, 기본 6, 두 방식 공통):
  침식 커널 반지름 `k = round(단축길이 × 침식%/100)` 로 마스크를 `cv2.erode`(타원 커널 `2k+1`). 목적은 **케이스 외곽선(배경과의 경계)** 을 지지영역에서 빼는 것 — 이 외곽선은 앞/뒤가 대칭이라 방향 단서가 안 되면서 에지/밝기 무게중심만 흔들기 때문이다.
  - 값이 **작으면** 외곽선 에지가 새어들어 방향이 흔들린다 → 키운다.
  - 값이 **크면** 내부 이음선/특징까지 깎여 신뢰도가 떨어진다 → 줄인다.
  - `k < 1`(=0%)이면 침식 생략. 침식 결과가 20픽셀 미만으로 너무 작아지면 **원본 마스크로 되돌린다**(작은 객체 안전장치).

- **에지 임계%** (`opening_thr_spin`, 범위 0–95, 기본 0=사용 안 함, **이음선 방식 전용**):
  약한 텍스처 노이즈(그림자·잔무늬)를 버리고 **강한 물리 에지(이음선)만** 남기는 백분위 임계.
  - **주의(중요):** 임계는 **개별 객체 지지영역이 아니라 캡처 이미지 전체**의 에지 크기 분포에서 잡는다(`np.percentile(grad, 임계%)`). 즉 임계% = 70 이면 **이미지 전역 에지의 상위 30%** 만 남기고 그 미만은 0. 지지영역 잘라내기는 이후 단계(침식)에서 하므로, 이 임계는 "전역적으로 약한 에지 제거" 로 이해하면 된다.
  - 배경이 복잡하거나 케이스 표면에 잔무늬가 많아 방향이 흔들릴 때 60~80으로 올린다. 이음선이 원래 약하면 오히려 이음선까지 지워지니 과하게 올리지 않는다.
  - 밝기 방식에는 적용되지 않는다.

- **방향 반전** (`opening_invert_chk`, 두 방식 공통):
  추정 벡터를 180° 뒤집는다. 각 방식의 "부호 의미"(위 표)가 **실제 제품과 반대로 나오는 경우**를 위한 수동 보정. 특히 밝기 방식은 밝은 쪽이 힌지냐 립이냐가 제품마다 달라 이 토글로 맞춘다.

> **신뢰도(confidence):** 단축 방향 오프셋을 단축 반길이로 나눈 비(0~1). **0.02 미만**(코드 고정 상수, 사용자 조정 아님)이면 비대칭이 뚜렷하지 않다는 뜻 → 상태바에 "신뢰도 낮음 N개" 경고. 인쇄 그래픽이 이음선보다 강하거나 힌지 금속이 크면 부호가 흔들릴 수 있다. **방식·침식%·에지 임계%·반전을 실제 케이스로 바꿔가며 가장 잘 맞는 조합을 찾는다** — 이음선이 안 맞으면 내부 밝기 비대칭을, 부호가 반대면 반전을 쓴다.

> **왜 밝기에서 최솟값을 빼나:** 무게중심은 가중치의 **상대적 분포**로 정해진다. 밝기를 원본(예 배경 ~128) 그대로 쓰면 큰 상수 baseline이 무게중심을 지지영역 기하중심 쪽으로 끌어당겨 **비대칭 신호를 눌러버린다**. `gray − min(gray)` 로 baseline을 없애면 어두운 배경부의 가중치가 0에 가까워져 밝은 특징의 치우침이 그대로 살아 **민감도가 올라간다**.

OBB·여는 방향은 새 검출/캡처 시 자동으로 지워진다(이전 결과가 새 화면에 남지 않도록).

---

## 4. 객체별 3D 포즈 추정

여기가 빈 픽킹의 핵심이다. 각 bbox에 대해 두 가지를 구한다:

- **중심점 (center)**: 어디로 다가갈 것인가
- **법선 (normal)**: 어떻게 다가갈 것인가

[bin_picking_tab.py:1129](../bin_picking_tab.py#L1129) `_compute_pick_poses()`.

### 4.1 중심점: bbox 안 3D 점들의 좌표 median

```python
crop_region = self.current_xyz[iy1:iy2, ix1:ix2].reshape(-1, 3)
valid_mask = ~np.any(np.isnan(crop_region), axis=1)
crop = crop_region[valid_mask]
if len(crop) < 20:
    continue   # 유효 점 너무 적음, 스킵
center = np.median(crop, axis=0)
```

**왜 median인가? 평균(mean)이 아니라?**

- bbox 안에는 물체 표면뿐만 아니라 배경(예: 옆 물체 일부, 박스 바닥)도 섞일 수 있다.
- 평균은 outlier(예: 갑자기 멀리 잡힌 점)에 민감하다 — 한 점만 뚝 떨어져도 중심이 움직인다.
- median은 이런 outlier에 **강건(robust)**하다. bbox 안 점의 절반 이상이 진짜 물체 표면이라면 median은 표면 위에 있다.

### 4.2 법선: 중심 주변 패치의 Zivid normals 평균

이 부분이 미묘하다. Zivid는 픽셀별 법선을 이미 제공하지만, **bbox 전체 평균은 안 된다** — 배경 픽셀의 법선이 섞여서 결과가 흐려지기 때문.

해법: **중심을 2D로 투영한 픽셀 주변의 작은 패치**(31×31)만 평균.

[bin_picking_tab.py:1171](../bin_picking_tab.py#L1171):

```python
# 중심점을 2D 픽셀로 투영
if fx is not None and center[2] > 0:
    center_px = int(round(fx * center[0] / center[2] + cx_i))
    center_py = int(round(fy * center[1] / center[2] + cy_i))
```

이 식은 **핀홀 카메라 모델**의 표준 투영식:
```
u = fx · X/Z + cx
v = fy · Y/Z + cy
```
3D 점을 영상 평면으로 투영해서 그 픽셀이 이미지 어디에 찍히는지 구한다.

그런 다음 그 픽셀 주변 31×31 영역의 normals를 평균:

```python
NORMAL_PATCH_RADIUS = 15  # → 31x31 (약 900 픽셀)

patch_n = self.current_normals[py1:py2, px1:px2].reshape(-1, 3)
valid_n = patch_n[~np.any(np.isnan(patch_n), axis=1)]
if len(valid_n) >= 3:
    mean_n = valid_n.mean(axis=0)
    normal = mean_n / np.linalg.norm(mean_n)
```

> **왜 `bbox 중심 픽셀`이 아니라 `3D 중심을 다시 투영한 픽셀`인가?**  
> bbox 기하학적 중심은 물체 표면이 아닐 수 있다 (예: 물체가 한쪽으로 치우쳐 들어있는 경우). 3D 중심(median)은 표면 위에 있을 확률이 높으므로, 이를 다시 픽셀로 투영하면 정확히 표면 점을 찾을 수 있다.

### 4.3 Fallback: SVD 평면 피팅

Zivid normals가 NaN이거나 카메라 설정상 normals를 못 가져왔을 때 쓰는 대비책. 중심 주변 XYZ 점들에 SVD로 평면을 피팅해서 법선을 구한다.

[bin_picking_tab.py:1227](../bin_picking_tab.py#L1227):

```python
@staticmethod
def _svd_normal(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[-1]  # 가장 작은 특이값 방향 = 평면 법선
```

(SVD가 왜 법선을 주는지는 [hand_eye_calibration.md](hand_eye_calibration.md#41-각-포즈에서-체커보드-자세-추정-t_target2cam) 4.1절 참고.)

### 4.4 법선 방향 정규화

법선은 두 방향이 가능하다 (`+n`과 `-n`). 픽킹용으로는 **카메라(또는 로봇)를 향하는 쪽**으로 통일해야 한다.

```python
if normal[2] > 0:
    normal = -normal
```

카메라 좌표계에서 +Z는 카메라가 보는 방향. 표면이 카메라 쪽을 향하려면 법선의 Z 성분이 음수여야 한다. 양수면 뒤집는다.

---

## 5. 카메라 좌표 → 로봇 좌표 변환

이 단계에서 **hand-eye calibration 결과 `T_cam2base`(또는 `T_cam2gripper`)**가 쓰인다.

[bin_picking_tab.py:1281](../bin_picking_tab.py#L1281):

### 5.1 Eye-to-Hand 모드

카메라가 외부에 고정 → `T_cam2base`가 곧 카메라→로봇 베이스 변환.

```python
center_cam = np.array(obj["center"])
normal_cam = np.array(obj["normal"])

p_h = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0])  # 동차 좌표
center_base = (self.T_calib @ p_h)[:3]
normal_base = self.T_calib[:3, :3] @ normal_cam   # 회전만 적용
```

- **위치**: 4×4 변환 행렬 × 동차 좌표(끝에 1 붙임).
- **법선**: 회전만 적용. 법선은 방향 벡터지 위치가 아니므로 평행이동을 더하면 안 된다.

### 5.2 Eye-in-Hand 모드

카메라가 로봇 손에 붙어있음 → 변환에 현재 로봇 자세도 포함되어야 한다.

```
T_cam2base = T_gripper2base @ T_cam2gripper
```

[bin_picking_tab.py:1289](../bin_picking_tab.py#L1289):

```python
cur_tcp = self.main.robot.get_tcp_position()
T_g2b = tcp_to_homogeneous(cur_tcp)

center_base = (T_g2b @ self.T_calib @ p_h)[:3]
normal_base = T_g2b[:3, :3] @ self.T_calib[:3, :3] @ normal_cam
```

여기서 `self.T_calib`는 `T_cam2gripper`이고, 현재 그리퍼의 베이스 기준 자세 `T_g2b`를 곱해서 최종 카메라→베이스 변환을 만든다.

> **두 모드 모두 결과는 똑같이** 베이스 좌표계의 `(center, normal)`. 이 둘만 있으면 다음 단계로 넘어갈 수 있다.

---

## 6. 로봇 TCP 자세 계산 (수직 접근)

이제 베이스 좌표계의 `target_base`(어디로)와 `normal_base`(어떻게)가 있다. 이걸 로봇이 이해할 수 있는 TCP 자세 `(x, y, z, A, B, C)`로 바꿔야 한다.

구현은 [calibration.py:507](../calibration.py#L507) `compute_approach_pose()`.

### 6.1 핵심 아이디어

- **Tool +Z 축**을 표면 법선의 **반대 방향**으로 정렬한다.
- "Tool +Z 반대"가 곧 표면을 찌르는 방향 → 표면에 수직으로 다가가게 됨.
- Tool X/Y 축은 자유롭게 정할 수 있다 (회전축이 한 자유도 남음). 현재 TCP의 X축에 가장 가깝게 골라서 자세 변화를 최소화.

### 6.2 단계별 계산

```python
normal_base = normal_base / np.linalg.norm(normal_base)

# Tool +Z = -법선 (표면을 향함)
new_z = -normal_base
```

다음으로 Tool X축을 정한다. **현재 TCP의 X축을 평면(new_z에 수직인 평면)에 투영**해서 사용한다 — 이렇게 하면 자세가 조금만 바뀌어 사람이 보기에 자연스럽다.

```python
cur_T = tcp_to_homogeneous(current_tcp)
cur_x = cur_T[:3, 0]  # 현재 TCP의 X축

# 현재 X축에서 new_z 방향 성분을 제거 → new_z에 수직인 부분만 남김
new_x = cur_x - np.dot(cur_x, new_z) * new_z
new_x = new_x / np.linalg.norm(new_x)
```

Tool Y축은 X와 Z의 외적으로 결정 (오른손 좌표계 유지):
```python
new_y = np.cross(new_z, new_x)
```

이제 회전 행렬 완성:
```python
R_new = np.column_stack([new_x, new_y, new_z])
```

그리고 4×4로 묶어 KUKA TCP 형식으로 변환:
```python
T_new = np.eye(4)
T_new[:3, :3] = R_new
T_new[:3, 3] = target_base

return homogeneous_to_tcp(T_new)   # → {"x", "y", "z", "a", "b", "c"}
```

### 6.3 엣지 케이스

현재 X축이 new_z와 거의 평행한 상황에서는 위 식이 0벡터를 만든다. 그땐 Y축을 fallback으로 쓰고, 그것도 안 되면 월드 X를 쓴다 ([calibration.py:540](../calibration.py#L540)):

```python
if nx_norm < 1e-6:
    cur_y = cur_R[:, 1]
    new_x = cur_y - np.dot(cur_y, new_z) * new_z
    nx_norm = np.linalg.norm(new_x)
    if nx_norm < 1e-6:
        new_x = np.array([1.0, 0.0, 0.0])
        new_x = new_x - np.dot(new_x, new_z) * new_z
```

이 fallback이 없으면 특정 자세에서 NaN 회전 행렬이 나와서 로봇이 미친 자세로 가려고 한다.

### 6.4 ABC unwrap — 큰 관절 회전 방지

`homogeneous_to_tcp`가 회전 행렬을 KUKA ZYX Euler로 분해할 때, **같은 자세인데도 ±180°wrap 경계를 넘으면 표현이 튄다**. 예: 현재 TCP `A=+170°` 인데 결과가 `A=-170°` 로 나오면 IK가 -340° 회전을 선택해 한쪽 관절이 한계까지 돌아가다 멈출 수 있다.

[calibration.py:`compute_approach_pose`](../calibration.py) 끝에 unwrap 추가:

```python
result = homogeneous_to_tcp(T_new)
for axis in ("a", "b", "c"):
    if axis in current_tcp:
        ref = float(current_tcp[axis])
        diff = (result[axis] - ref + 180.0) % 360.0 - 180.0
        result[axis] = ref + diff
return result
```

수식 의미: 결과 각도를 **현재 TCP에 가장 가까운 모듈로-360 표현**으로 조정. 같은 자세를 가리키는 두 표현 중 회전 변화가 작은 쪽 선택.

### 6.5 회전 변화량 표시 + 큰 회전 경고

객체 선택 직후 [bin_picking_tab.py:`_compute_rotation_change_deg`](../bin_picking_tab.py) 가 현재 TCP와 새 target 자세 사이의 회전 변화량을 axis-angle 로 계산:

```
R_diff = R_cur^T @ R_target
trace = (Tr(R_diff) - 1) / 2
angle_deg = acos(clip(trace, -1, 1)) · 180/π
```

상태바에 `회전변화 N°` 표시. **60° 초과 시 경고**(`⚠ 큰 회전 — PTP 권장`) 가 같이 뜸. 사용자가 LIN 대신 PTP 를 선택하거나 현재 TCP 자세를 미리 조정할 수 있게.

### 6.6 3D 뷰에 Tool 자세 시각화

법선 화살표 하나만으로는 그리퍼가 어느 방향으로 어떻게 회전해서 접근할지 보이지 않는다. 객체 선택 시 [bin_picking_tab.py:`_render_tcp_visualization`](../bin_picking_tab.py) 가 3D 뷰에 다음을 추가:

- **Tool 좌표축 3개**: X 빨강, Y 초록, Z 파랑 (각 50 mm)
- **Approach 지점**: Tool -Z 방향으로 offset 만큼 떨어진 주황 구
- **Approach → Target 경로선**: 주황 직선

좌표계 변환 주의: `target_pose` 는 **베이스 좌표계** 자세인데 3D 뷰는 **카메라 좌표계**다. 그대로 그리면 객체와 동떨어진 곳에 표시되므로 회전을 변환해야 한다:

```python
# Eye-to-Hand:
R_in_cam = T_calib[:3, :3].T @ R_target_base
# Eye-in-Hand:
R_in_cam = T_calib[:3, :3].T @ T_g2b[:3, :3].T @ R_target_base
```

위치는 `obj["center"]` (이미 카메라 좌표계 객체 중심)을 사용. 이 시각화 덕분에 매칭 직후 사용자가 그리퍼가 의도대로 접근할지 즉시 확인 가능.

---

## 7. 접근/철수(Approach/Retract) 모션

물체로 곧장 직선으로 다가가면 옆 물체와 충돌할 수 있다. 그래서 **3단계 모션**을 쓴다:

1. **Approach** — 목표 자세로 회전한 채로, 표면에서 N mm 떨어진 위치에 먼저 도착.
2. **Target** — Approach 위치에서 표면까지 직선(LIN)으로 정밀 접근.
3. **Retract** — Target에서 다시 Approach 위치로 직선 후퇴.

이렇게 하면 회전은 멀리서 끝내고, 표면 근처에서는 직선 운동만 하므로 충돌 위험이 적다.

### 7.1 Approach 위치 계산

[bin_picking_tab.py:1331](../bin_picking_tab.py#L1331) `_compute_approach_position()`:

```python
T = tcp_to_homogeneous(target)   # 목표 자세
z_axis = T[:3, 2]                # Tool +Z (표면을 향함)
target_pos = T[:3, 3]
approach_pos = target_pos - z_axis * offset_mm   # Tool -Z 방향으로 offset만큼 떨어진 위치
```

`offset_mm`은 사용자가 UI에서 설정 (기본값 50~100mm 정도).

### 7.2 KRL 큐에 3개 모션 추가

[bin_picking_tab.py:1424](../bin_picking_tab.py#L1424):

```python
slot1 = self.main.robot.add_move_ptp(ax, ay, az, p["a"], p["b"], p["c"])      # Approach (PTP, 빠른 이동)
slot2 = self.main.robot.add_move_lin(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])  # Target (LIN, 안전한 직선)
slot3 = self.main.robot.add_move_lin(ax, ay, az, p["a"], p["b"], p["c"])      # Retract (LIN)
```

세 모션이 큐에 한꺼번에 들어가면 KRL이 알아서 차례대로 실행한다 ([kuka_communication.md](kuka_communication.md) 3장).

> **주의**: 이 "Approach → Target → Retract" 는 **한 지점을 찍고 바로 빠지는** 동작이라, 진공/집기 그리퍼로 **실제로 물건을 집는 픽 사이클과는 다르다**. 물건을 집으려면 Target 도착 후 그리퍼를 작동시키고 물건을 든 채 빠져나와야 한다 — 다음 장의 픽 사이클을 참고.

---

## 7.5. 픽 사이클 (진공 그리퍼 — 잡기→놓기)

상용 빈 픽킹 프로그램의 표준 동작이다. 단순 "이동"과 달리 **모션 사이사이에 그리퍼 작동을 끼워 넣어야** 물건을 실제로 집는다:

```
① Approach  (PTP, 빠르게)   물체 위 offset mm 지점으로
② 하강      (LIN, 정밀)     물체 표면까지 직선
③ 진공 ON   + 흡착 대기(dwell)   빨아들일 시간을 준다
④ 상승      (LIN, 잡은 채)  다시 Approach 높이로
⑤ 놓기 이동 (PTP)           티칭해둔 Place 위치로
⑥ 진공 OFF  + 블로우 펄스    확실히 떨어뜨림
⑦ Home 복귀 (PTP, 선택)
```

### 왜 "모든 모션을 큐에 한꺼번에" 방식이 안 되나

7장의 단순 이동은 Approach/Target/Retract 3개를 KRL 큐에 한 번에 쌓고 KRL이 알아서 연속 실행했다. 그런데 픽 사이클은 **③에서 로봇이 ②까지 끝나고 멈춘 정확한 순간에** 진공을 켜야 한다. 모션을 한꺼번에 던지면 "언제 ②가 끝났는지" Python이 알 수 없어 진공 타이밍을 못 맞춘다.

그래서 [robot_control_mixin.py](../robot_control_mixin.py)는 **단계별 실행기(`_run_cycle` / `_cycle_tick`)** 를 쓴다 — `QTimer`로 150ms마다 상태를 확인하는 상태 머신:

1. 모션 스텝: 슬롯 **하나**만 KRL 큐에 넣고, 그 슬롯의 `robo_motion_type[slot]`이 0으로 돌아올 때까지(=완료) 다음 스텝으로 안 넘어감.
2. 진공 스텝: `set_vacuum()` 호출 ($OUT readback 으로 실제 적용 확인).
3. dwell 스텝: 지정 시간만큼 대기.

스텝은 `_build_pick_steps()`가 생성하고, `("move","ptp"|"lin",pose)` / `("vacuum",bool)` / `("blow",초)` / `("dwell",초)` / `("status",문구)` 형식이다.

### Place(놓기) 위치 티칭

Home 과 똑같은 방식으로, 로봇을 놓을 자리로 조그(jog)한 뒤 **"📍 놓기 위치 저장"** 을 누르면 현재 TCP 가 `main.place_pose` 에 저장된다 (탭 공용). 픽 사이클/시퀀스 픽이 여기에 내려놓는다.

### 두 가지 사용 방식

- **원버튼 픽** ("🤖 픽 실행"): 선택한 대상 1개를 즉시 픽 사이클 실행 (Home 복귀까지).
- **시퀀스 픽** ("➕ 픽 시퀀스 추가"): 여러 대상의 픽을 시퀀스 큐에 쌓아 한 번에 연속 실행 (§10). 이때 각 픽은 Home 복귀를 빼고, 필요하면 사용자가 "Home 추가"로 사이에 끼운다.

비상정지(Space/버튼)를 누르면 진행 중이던 픽/시퀀스 사이클도 즉시 중단되어 남은 스텝이 전송되지 않는다.

---

## 8. 안전장치

빈 픽킹은 가장 사고 위험이 큰 작업이다 (물리적 접촉이 의도된 작업이므로). 본 프로그램은 여러 단계 안전장치를 둔다:

### 8.1 Z 좌표 하한

[bin_picking_tab.py:1536](../bin_picking_tab.py#L1536) `_validate_z()`:

```python
def _validate_z(self, z: float) -> bool:
    z_min = self.z_min_spin.value()
    if z < z_min:
        QMessageBox.critical(...)  # 거부
        return False
    return True
```

작업대 표면 아래로 내려가는 명령은 큐에 추가되기 전에 차단.

### 8.2 AUT 모드 속도 제한

`is_auto_mode(self._current_mode)` (공통 헬퍼)가 True 면 50% 상한:

```python
def _effective_speed(self, requested: int) -> int:
    if self._is_aut_mode():
        return min(requested, 50)   # AUT/AUT_EXT에서는 50% 상한
    return requested
```

AUT 모드 감지는 [kuka_robot.py:`normalize_robot_mode`/`is_auto_mode`](../kuka_robot.py) 공통 함수로 통일되어 있다 — 이전엔 두 탭에서 다른 substring/exact 매칭을 써서 `#EXT` 같은 경우 한쪽 탭에서만 속도 제한이 걸리는 안전 불일치가 있었으나, 정규화/판정 함수를 한 곳에 두어 해결.

### 8.3 시퀀스 큐 approach 지점 Z 안전 검증

`_start_sequence` 가 각 액션의 target Z 뿐 아니라 **계산된 approach 지점 Z** 도 `_validate_z` 로 검증 (Tool +Z 가 옆/위를 향하면 approach 가 바닥 한계 아래로 내려갈 수 있음):

```python
for action in self.user_queue:
    if not self._validate_z(action["target"]["z"]):
        return
    if action["type"] == "object_move" and action.get("use_approach", True):
        ax, ay, az = self._compute_approach_position(
            action["target"], action.get("approach_dist", 50)
        )
        if not self._validate_z(az):
            return
```

`_execute_move` 단일 모션 경로도 같은 검증 적용.

T1 모드(데드맨 스위치 필요, 250mm/s 이하)와 달리 AUT는 사람이 잡지 않은 채로 풀속도까지 갈 수 있다. 그래서 소프트웨어 단에서도 한 겹 더 막는다.

### 8.4 비상정지 / 큐 비우기

이동 중 위험 시 [bin_picking_tab.py:1463](../bin_picking_tab.py#L1463) `_emergency_stop()`:

```python
self.main.robot.emergency_stop()      # robo_scram = TRUE → KRL이 즉시 BRAKE F
self.main.robot.clear_queue()          # 큐 모든 슬롯 type = 0
```

KRL의 인터럽트 핸들러에 `RESUME`이 들어있어서, 비상정지 해제 시 현재 모션을 취소하고 깔끔히 LOOP로 돌아간다 ([kuka_communication.md](kuka_communication.md) 3.4절).

### 8.5 사용자 확인 다이얼로그

이동 직전에 목표 위치/속도/자세를 다 보여주고 Yes/No 확인을 받는다 ([bin_picking_tab.py:1374](../bin_picking_tab.py#L1374)). 실수로 버튼 잘못 눌렀을 때의 마지막 방어선.

---

## 9. 전체 흐름 요약

```
[입력]
사용자 → "캡처" 버튼
         ↓
Zivid → 2D 이미지 + 3D XYZ + 법선 맵 + intrinsics
         ↓
사용자 → ROI 드래그 (선택)
사용자 → "검출" 버튼
         ↓
[객체 검출]
RF-DETR → 검출 리스트 (bbox + class + 선택적 mask)
         ↓
ROI 필터링 (bbox 중심의 3D 좌표 검사)
         ↓
[3D 포즈 추정]   각 bbox에 대해:
   center  = median(bbox 안 유효 XYZ)
   center_pixel = 핀홀 모델로 다시 투영
   normal  = center 주변 31×31 패치의 Zivid normals 평균
            (없으면 SVD 평면 피팅)
   법선 부호: 카메라 향하도록 정규화
         ↓
[좌표 변환]   사용자가 객체 클릭:
   (Eye-to-Hand)   center_base = T_calib @ center_cam
                   normal_base = R_calib @ normal_cam
   (Eye-in-Hand)   center_base = T_g2b @ T_calib @ center_cam
                   ...
         ↓
[TCP 자세 계산]   compute_approach_pose():
   Tool +Z = -normal_base  (표면을 향함)
   Tool X  = current_TCP_X를 (Z에 수직 평면)으로 투영
   Tool Y  = Z × X
   → R_new (3×3) → R + target_base → 4×4 → KUKA TCP {x,y,z,A,B,C}
         ↓
[모션 큐]   사용자가 "이동" 버튼:
   Approach 위치 = target - Tool_Z * offset
   robot.add_move_ptp(approach)   →  큐 슬롯 N
   robot.add_move_lin(target)     →  큐 슬롯 N+1
   robot.add_move_lin(retract)    →  큐 슬롯 N+2
         ↓
[KRL]  ext_move.src LOOP가 슬롯 차례대로 실행
       → 실제 모터 구동
```

---

## 10. 보너스: 시퀀스 큐 (배치 픽킹)

여러 객체를 한 번에 자동으로 처리하고 싶다면, 사용자가 객체를 차례로 추가해 "시퀀스"를 만들어 한 번에 실행하는 기능이 있다 ([bin_picking_tab.py:1554](../bin_picking_tab.py#L1554) `_enqueue_object_move()` 등).

내부적으로는:
1. 각 추가 시 **현재 프레임 기준의 Approach/Target/Retract 자세**를 계산해 user_queue 리스트에 저장 (단순 좌표 3개).
2. "시작" 버튼을 누르면 user_queue를 순회하며 KRL 큐에 모션을 채운다.
3. 사이사이 Home 위치 복귀를 끼워넣을 수도 있다.

> **주의**: 시퀀스의 자세는 **추가 시점**의 카메라 영상으로 계산된 값이다. 그 사이에 박스 안 물체가 흔들리거나 움직이면 좌표가 어긋난다. 실제 운영에서는 한 사이클마다 다시 캡처+검출을 해서 좌표를 갱신하는 게 안전하다.

---

## 10.5. 공통 로봇 제어 Mixin

빈 픽킹 탭과 CAD 매칭 탭은 거의 동일한 "로봇 이동 제어 + 시퀀스 큐 + 안전" 코드 약 400줄을 공유했었는데, 이걸 [robot_control_mixin.py:`RobotControlMixin`](../robot_control_mixin.py) 으로 추출해 두 탭이 상속한다:

```python
class BinPickingTab(RobotControlMixin, QWidget):
    SEQ_OBJECT_NOUN = "객체"     # 시퀀스 라벨용 (기본값)
    ...

class CADMatchingTab(RobotControlMixin, QWidget):
    SEQ_OBJECT_NOUN = "인스턴스"  # CAD 매칭은 "인스턴스"라 부름
    ...
```

Mixin 안에 들어간 메서드 (19개):

- 속도: `_on_speed_changed`, `_apply_speed_now`
- 안전: `_validate_z`, `_effective_speed`, `_is_aut_mode`, `_emergency_stop`, `_emergency_stop_release`
- Home: `_set_home_to_current`, `_move_to_home`, `_clear_motion_queue`
- 시퀀스 큐: `_refresh_action_list`, `_enqueue_object_move`, `_enqueue_home_to_sequence`, `_remove_selected_action`, `_clear_user_queue`, `_start_sequence`, `_send_action_to_krl_queue`
- 진공 그리퍼 / 픽 사이클: `_build_vacuum_row` (진공+픽 버튼 2행 생성), `_set_vacuum_ui`, `_vacuum_blow_ui`, `_set_place_to_current`, `_execute_pick_cycle`, `_enqueue_pick`, 단계별 실행기 `_run_cycle`/`_cycle_tick`/`_build_pick_steps`/`_expand_action_to_steps`
- 기타: `_on_robot_connected`, `_compute_approach_position` (staticmethod)

`_refresh_mode_display` 만 탭별 표시 스타일이 달라서 Mixin 밖에 남았다 (logic은 공통 `is_auto_mode` 사용).

이렇게 통합한 덕분에 e-stop 해제 동작 같은 안전 관련 수정이 한 곳에서 끝나고 두 탭이 자동으로 동일하게 동작 — 이전엔 두 탭이 따로 수정되다 안전 동작이 어긋난 적이 있었음.

---

## 11. 한계와 개선 여지

이 시스템이 다루지 못하는 / 단순화한 부분:

- **수직 접근 가정**: 모든 객체에 표면 법선 수직 방향으로 다가간다. 더 복잡한 그리퍼/객체 형태에서는 객체 자체의 자세(orientation)도 같이 추정해야 한다 (6DoF pose estimation, 예: PVN3D, FoundationPose 등).
- **형상 매칭 없음**: RF-DETR는 "물체가 거기 있다"만 알려준다. 실제로 그리퍼가 잡기 좋은 손잡이/평면이 어딘지는 본 시스템이 알지 못한다.
- **충돌 회피 없음**: Approach → Target 직선 운동이 다른 물체를 안 건드린다는 보장은 없다. 점유 격자(occupancy grid) 기반 모션 플래너를 붙이면 안전성이 크게 올라간다.
- **반복 가능성**: 한 번 검출한 후 시퀀스를 시작하면 그 사이 박스 내부 변화를 반영하지 못한다 (10장 참고).

학습 단계에서 이 한계들이 어떻게 해결되는지 알아두면 다음 단계 시스템을 설계할 때 도움이 된다.

---

## 12. 참고

- [bin_picking_tab.py](../bin_picking_tab.py) — 빈 픽킹 탭 전체 구현
- [calibration.py:507](../calibration.py#L507) `compute_approach_pose()` — 수직 접근 자세 계산
- [zivid_camera.py](../zivid_camera.py) — Zivid 캡처 / normals / intrinsics
- [hand_eye_calibration.md](hand_eye_calibration.md) — `T_calib`이 어떻게 만들어졌는지
- [kuka_communication.md](kuka_communication.md) — `add_move_*` 호출이 실제 로봇까지 어떻게 전달되는지
- 핀홀 카메라 모델: `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` (OpenCV docs 참고)
