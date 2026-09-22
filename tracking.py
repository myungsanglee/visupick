"""
Object Tracking (순수 모듈 — Qt 무의존)
=======================================
검출 결과를 프레임 사이로 이어 붙여 **같은 물체에 같은 ID**를 부여한다.
base_camera.py / object_detector.py 와 같은 발상: 인터페이스를 두고 구현체를 갈아끼운다.

**검출기 비종속이다.** 입출력이 이 저장소의 검출 표준 포맷(object_detector 참고)이라
SAM3·RF-DETR 뿐 아니라 앞으로 추가될 어떤 검출기든 그대로 동작한다:

    detections = [{"bbox":[x1,y1,x2,y2], "confidence":float, "class_id":int,
                   "class_name":str, "mask":(H,W) bool 또는 없음}, ...]
    tracked    = tracker.update(detections, frame=None)   # 각 항목에 "track_id" 추가

구현체 (셋 다 칼만 + IoU/GIoU + 헝가리안 할당을 공유하고, 연관 전략만 다르다):
  - SortTracker      : 원조 SORT. **1단계 연관만** — 임계 이상 검출 전부를 한 번에 맞춘다.
                       가장 단순해서 비교 기준(baseline)으로 쓴다.
  - ByteTrackTracker : 2단계 연관(고신뢰 → 저신뢰). 검출이 잠깐 흔들려도 트랙을 잇는다.
  - BotSortTracker   : ByteTrack + BoT-SORT 개선 (칼만 상태 x,y,w,h + GIoU + CMC 옵션).
                       **ReID(외형 임베딩)는 제외** — 딥러닝 모델이 필요하고, 우리처럼
                       종류가 적고 생김새가 비슷한 화장품 케이스에서는 이득이 적다.

왜 roboflow/trackers 라이브러리를 안 쓰나
----------------------------------------
그 라이브러리는 **numpy>=2.0.2** 를 요구하는데, 이 저장소의 주력 검출기 SAM 3 는
**numpy<2** 를 요구한다(requirements.txt 참고). 게다가 opencv-python 을 끌어와
opencv-contrib 와 충돌시켜 cv2.ppf_match_3d(CAD 매칭 PPF)와 cv2.ximgproc.thinning
(표면 추적 선 검출)을 망가뜨린다 — 실제로 설치해 보고 확인했다. 버전 조정으로 풀리는
문제가 아니라서, 필요한 알고리즘만 직접 구현했다(scipy/opencv/numpy 만 사용, 새 의존성 0).
나중에 제약이 풀리면 ObjectTracker 를 구현하는 백엔드를 추가로 끼우면 된다.

좌표 규약: bbox 는 이미지 픽셀 xyxy. 칼만 상태는 중심(cx,cy)+크기(w,h)와 그 속도.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================
# 기하 헬퍼
# ============================================================


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 bbox 집합의 IoU 행렬 (len(a), len(b)). 빈 입력이면 (len(a), len(b)) 영행렬."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1, 4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def giou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GIoU 행렬 (−1~1). IoU 와 달리 **안 겹치는 쌍도 거리로 구분**한다.

    IoU 는 떨어져 있으면 전부 0 이라 "조금 빗나감"과 "완전히 딴 데"를 구분하지 못한다.
    빠르게 움직여 예측과 검출이 스치듯 어긋난 프레임에서 GIoU 가 더 잘 잇는다.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1, 4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = np.maximum(area_a + area_b - inter, 1e-9)
    iou = inter / union
    # 두 상자를 모두 감싸는 최소 사각형(convex hull)
    cx1 = np.minimum(a[..., 0], b[..., 0])
    cy1 = np.minimum(a[..., 1], b[..., 1])
    cx2 = np.maximum(a[..., 2], b[..., 2])
    cy2 = np.maximum(a[..., 3], b[..., 3])
    c_area = np.maximum(np.clip(cx2 - cx1, 0, None) * np.clip(cy2 - cy1, 0, None), 1e-9)
    return iou - (c_area - union) / c_area  # 멀수록 −1 쪽으로


def _assign(cost: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """헝가리안 최적 할당 후 비용이 임계보다 큰 쌍은 버린다.

    탐욕적 매칭 대신 헝가리안을 쓰는 이유: 물체 두 개가 스쳐 지나갈 때 탐욕적으로는
    "각자 가장 가까운 쪽"을 집어 **ID 가 서로 뒤바뀌는** 경우가 생긴다. 전체 비용 합을
    최소화하면 그런 교차 오류가 크게 줄어든다.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    r_idx, c_idx = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(r_idx, c_idx) if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return matches, [r for r in range(rows) if r not in matched_r], [c for c in range(cols) if c not in matched_c]


# ============================================================
# 칼만 필터 (등속 모델, 상태 = [cx, cy, w, h, vx, vy, vw, vh])
# ============================================================


class _KalmanBox:
    """bbox 전용 8상태 등속 칼만 필터.

    SORT 원본은 상태를 (cx, cy, 넓이 s, 종횡비 a) 로 두는데, BoT-SORT 는 **(cx, cy, w, h)**
    로 바꾼다. 종횡비를 상수처럼 다루면 물체가 회전하거나 가려져 한쪽만 보일 때 폭·높이가
    함께 왜곡되기 때문. 여기서도 BoT-SORT 쪽을 따른다.
    """

    def __init__(self, bbox: Sequence[float], std_pos: float = 0.05, std_vel: float = 0.00625):
        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._std_pos = float(std_pos)
        self._std_vel = float(std_vel)
        # 초기 불확실성: 속도는 전혀 모르므로 크게 잡는다 (위치보다 10배)
        s = np.array(
            [
                2 * std_pos * h,
                2 * std_pos * h,
                2 * std_pos * h,
                2 * std_pos * h,
                10 * std_vel * h,
                10 * std_vel * h,
                10 * std_vel * h,
                10 * std_vel * h,
            ]
        )
        self.P = np.diag(np.square(s))
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # 등속: 위치 += 속도
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

    def _noise(self) -> Tuple[np.ndarray, np.ndarray]:
        """잡음 크기를 **물체 높이에 비례**시킨다 — 멀리 있어 작게 보이는 물체는 픽셀 단위
        움직임도 작으므로, 고정 잡음을 쓰면 크고 작은 물체에 같은 기준이 적용돼 버린다."""
        h = max(float(self.x[3]), 1.0)
        q = np.square(np.array([self._std_pos * h] * 4 + [self._std_vel * h] * 4))
        r = np.square(np.array([self._std_pos * h] * 4))
        return np.diag(q), np.diag(r)

    def predict(self) -> np.ndarray:
        Q, _ = self._noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        self.x[2] = max(self.x[2], 1.0)  # 폭/높이가 0 이하로 발산하지 않게
        self.x[3] = max(self.x[3], 1.0)
        return self.bbox

    def update(self, bbox: Sequence[float]) -> None:
        _, R = self._noise()
        z = np.array(_xyxy_to_cxcywh(bbox), dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph 형태 (대칭·양정부호 유지)

    def apply_affine(self, M: np.ndarray) -> None:
        """카메라가 움직였을 때 상태를 그 변환만큼 옮긴다 (CMC). M 은 2×3 affine."""
        cx, cy, w, h = self.x[:4]
        pt = M[:, :2] @ np.array([cx, cy]) + M[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) or 1.0
        self.x[0], self.x[1] = pt
        self.x[2] *= scale
        self.x[3] *= scale
        self.x[4:6] = M[:, :2] @ self.x[4:6]  # 속도는 회전/스케일만 (평행이동 제외)
        self.x[6:8] *= scale

    @property
    def bbox(self) -> np.ndarray:
        return np.asarray(_cxcywh_to_xyxy(self.x[:4]), dtype=np.float64)


def _xyxy_to_cxcywh(b: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _cxcywh_to_xyxy(s: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in s[:4])
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


# ============================================================
# 트랙
# ============================================================


class Track:
    """추적 중인 물체 하나. 상태 전이: tentative → confirmed → (미검출 누적) → 삭제."""

    _next_id = 1

    def __init__(self, det: Dict, min_hits: int):
        self.kf = _KalmanBox(det["bbox"])
        self.track_id: Optional[int] = None  # confirmed 될 때 부여 (깜빡이는 오검출에 번호 낭비 방지)
        self.class_id = det.get("class_id", 0)
        self.class_name = det.get("class_name", "")
        self.confidence = float(det.get("confidence", 0.0))
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self._min_hits = min_hits
        if min_hits <= 1:
            self._confirm()

    def _confirm(self):
        if self.track_id is None:
            self.track_id = Track._next_id
            Track._next_id += 1

    @property
    def confirmed(self) -> bool:
        return self.track_id is not None

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Dict):
        self.kf.update(det["bbox"])
        self.confidence = float(det.get("confidence", self.confidence))
        if det.get("class_name"):
            self.class_name = det["class_name"]
            self.class_id = det.get("class_id", self.class_id)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self._min_hits:
            self._confirm()

    @staticmethod
    def reset_ids():
        """ID 카운터 초기화 (테스트/새 세션용). 트래커 생성 시 자동 호출되지는 않는다 —
        여러 트래커를 동시에 써도 ID 가 겹치지 않아야 하므로 전역 카운터를 공유한다."""
        Track._next_id = 1


# ============================================================
# 트래커 인터페이스 + 구현
# ============================================================


class ObjectTracker(ABC):
    """트래커 공통 인터페이스. 새 구현(외부 라이브러리 백엔드 포함)은 이걸 상속한다."""

    name = "tracker"

    @abstractmethod
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """검출 목록 → **track_id 가 붙은** 검출 목록.

        frame 은 카메라 모션 보정(CMC)처럼 영상이 필요한 구현만 사용한다 (없어도 동작).
        반환 항목은 입력 dict 의 얕은 복사 + "track_id" — 입력을 훼손하지 않는다.
        """

    @abstractmethod
    def reset(self) -> None:
        """모든 트랙 제거 (새 촬영 시작 등)."""


class ByteTrackTracker(ObjectTracker):
    """ByteTrack — 고신뢰 검출로 먼저 잇고, 남은 트랙을 **저신뢰 검출로 한 번 더** 잇는다.

    저신뢰 검출을 그냥 버리면 물체가 잠깐 가려지거나 흐려질 때 트랙이 끊겨 ID 가 바뀐다.
    2단계 연관은 그런 프레임을 "위치가 맞으면 살려 두는" 방식으로 메운다 — 새 ID 를
    만드는 데는 여전히 고신뢰 검출만 쓰므로 오검출이 트랙으로 승격되지는 않는다.
    """

    name = "ByteTrack"

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.2,
        second_match_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        class_aware: bool = True,
        second_stage: bool = True,
    ):
        """
        high_thresh       : 이 이상이면 '고신뢰' — 1단계 연관 + 새 트랙 생성에 사용
        low_thresh        : 이 미만 검출은 아예 버림
        match_thresh      : 1단계 매칭 **최소 IoU**. 원 논문(ByteTrack)의 iou_distance
                            임계 0.8 과 같은 값 — 거기선 '1−IoU ≤ 0.8' 이므로 IoU ≥ 0.2 다.
                            너무 높이면(예: 0.8) 프레임당 조금만 움직여도 매칭이 끊겨
                            매 프레임 새 ID 가 생긴다.
        second_match_thresh : 2단계(저신뢰) 매칭 최소 IoU. 1단계보다 **더 엄격**하다 —
                            저신뢰 검출은 위치가 덜 믿음직해서 기하학적으로 더 확실할
                            때만 잇는다 (원 논문도 0.5).
        max_age           : 미검출 몇 프레임까지 트랙을 유지할지 (가림 대응)
        min_hits          : 몇 번 연속 잡혀야 ID 를 부여할지 (깜빡이는 오검출 배제)
        class_aware       : True 면 **다른 class 끼리는 매칭 금지** (쉼표 다중 프롬프트로
                            여러 종류를 검출할 때 종류가 뒤바뀌지 않게)
        second_stage      : 저신뢰 검출로 한 번 더 잇는 2단계 연관 사용 여부.
                            False 면 SORT 와 같은 1단계 연관이 된다 (SortTracker 가 이걸 끈다).
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.class_aware = class_aware
        self.second_stage = second_stage
        self.tracks: List[Track] = []

    # -- 하위 클래스가 바꿔 끼우는 지점 --
    def _similarity(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        return iou_matrix(track_boxes, det_boxes)

    def _compensate(self, frame: Optional[np.ndarray]) -> None:
        return None

    def reset(self) -> None:
        self.tracks = []

    def _cost(self, tracks: List[Track], dets: List[Dict]) -> np.ndarray:
        """비용 = 1 − 유사도. class_aware 면 다른 종류끼리는 무한대로 막는다."""
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        sim = self._similarity(np.array([t.bbox for t in tracks]), np.array([d["bbox"] for d in dets], dtype=np.float64))
        cost = 1.0 - sim
        if self.class_aware:
            t_cls = np.array([t.class_id for t in tracks]).reshape(-1, 1)
            d_cls = np.array([d.get("class_id", 0) for d in dets]).reshape(1, -1)
            cost = np.where(t_cls == d_cls, cost, 1e6)
        return cost

    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        detections = list(detections or [])
        self._compensate(frame)
        for t in self.tracks:
            t.predict()

        high = [d for d in detections if float(d.get("confidence", 1.0)) >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= float(d.get("confidence", 1.0)) < self.high_thresh]

        # 1단계: 모든 트랙 × 고신뢰 검출.
        # 주의: m1/un_t 는 **이 시점의 self.tracks 인덱스**다. 아래에서 새 트랙을 append
        # 하기 전에 rest 를 떠 두어야 인덱스가 어긋나지 않는다.
        m1, un_t, un_hi = _assign(self._cost(self.tracks, high), 1.0 - self.match_thresh)
        for ti, di in m1:
            self.tracks[ti].update(high[di])

        # 2단계: 아직 못 이은 트랙 × 저신뢰 검출 (여기서는 새 트랙을 만들지 않는다)
        rest = [self.tracks[i] for i in un_t]
        m2: List[Tuple[int, int]] = []
        if self.second_stage and low:
            m2, _, _ = _assign(self._cost(rest, low), 1.0 - self.second_match_thresh)
            for ti, di in m2:
                rest[ti].update(low[di])

        # 남은 고신뢰 검출 → 새 트랙. min_hits=1 이면 이 프레임에 바로 확정되므로
        # 결과에도 포함돼야 한다 → 짝을 같이 모아 둔다.
        born: List[Tuple[Dict, Track]] = []
        for di in un_hi:
            tr = Track(high[di], self.min_hits)
            self.tracks.append(tr)
            born.append((high[di], tr))

        # 오래 못 본 트랙 정리
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 결과: **이번 프레임에 실제로 매칭된** confirmed 트랙만 내보낸다.
        # 예측만으로 만든 유령 상자를 내보내면 화면에 없는 물체가 잡힌 것처럼 보인다.
        # 순서는 입력 검출 순서를 따른다 (색 팔레트·테이블 순서가 흔들리지 않게).
        matched: List[Tuple[Dict, Track]] = [(high[di], self.tracks[ti]) for ti, di in m1]
        matched += [(low[di], rest[ti]) for ti, di in m2]
        matched += born
        order = {id(d): i for i, d in enumerate(detections)}
        matched.sort(key=lambda pair: order.get(id(pair[0]), 1 << 30))

        out: List[Dict] = []
        for det, track in matched:
            if not track.confirmed:
                continue
            item = dict(det)
            item["track_id"] = track.track_id
            out.append(item)
        return out

    def __repr__(self) -> str:
        return f"{self.name}(tracks={len(self.tracks)})"


class SortTracker(ByteTrackTracker):
    """SORT (Simple Online and Realtime Tracking) — 가장 단순한 형태.

    칼만으로 예측하고 **IoU + 헝가리안으로 한 번만** 맞춘다. ByteTrack 의 저신뢰 2단계
    연관이 없으므로, 검출이 한 프레임 흐려지면 그 트랙은 그냥 끊긴다(그 대신 저신뢰
    오검출에 끌려갈 일도 없다). 세 구현 중 가장 보수적이라 **비교 기준**으로 쓴다.

    원 논문은 검출을 신뢰도로 나누지 않으므로 high_thresh 를 낮게(0.1) 두어 사실상
    모든 검출을 1단계에 넣는다.

    ※ 칼만 상태는 이 모듈 공용인 (cx, cy, w, h) 를 쓴다 — 원 SORT 의 (u, v, s, r)
      (중심+넓이+종횡비) 와는 다르다. 종횡비를 상수처럼 다루면 물체가 회전하거나 일부만
      보일 때 폭·높이가 함께 왜곡되기 때문에, BoT-SORT 쪽 표현으로 통일했다.
    """

    name = "SORT"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("high_thresh", 0.1)  # 신뢰도로 나누지 않음 = 전부 1단계
        kwargs["second_stage"] = False
        super().__init__(*args, **kwargs)


class BotSortTracker(ByteTrackTracker):
    """BoT-SORT(ReID 제외) — ByteTrack 에 두 가지를 더한다.

    1. **GIoU 유사도** — 안 겹치는 쌍도 거리로 구분해, 빠르게 움직여 스치듯 어긋난
       프레임에서 더 잘 잇는다.
    2. **카메라 모션 보정(CMC, 옵션)** — 배경 특징점의 광학 흐름으로 프레임 간 affine 을
       추정해 트랙 예측을 그만큼 옮긴다.

    ⚠️ **CMC 기본값은 꺼짐**이다. 이 시스템은 카메라가 고정(eye-to-hand)이라 보정할
    움직임이 없어 이득은 없고 잡음만 들어온다. 나중에 카메라를 로봇 팔에 다는
    eye-in-hand 구성으로 가면 그때 켜면 된다(그 경우엔 프레임마다 시점이 바뀐다).

    칼만 상태 (cx,cy,w,h) 는 _KalmanBox 에서 이미 BoT-SORT 방식이라 별도 처리가 없다.
    ReID 는 넣지 않았다 — 딥러닝 모델이 추가로 필요하고, 생김새가 거의 같은 화장품
    케이스에서는 외형 임베딩이 오히려 헷갈리기 쉽다.
    """

    name = "BoT-SORT(w/o ReID)"

    def __init__(self, *args, use_cmc: bool = False, cmc_max_corners: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cmc = use_cmc
        self.cmc_max_corners = cmc_max_corners
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        super().reset()
        self._prev_gray = None

    def _similarity(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        """GIoU 를 **그대로** 쓴다 (0~1 로 옮기지 않는다).

        겹치는 쌍에서는 GIoU ≈ IoU 라 match_thresh(최소 IoU) 의미가 그대로 통하고,
        겹치는 후보들 사이에서는 '얼마나 어긋났는지'가 비용에 반영돼 **교차 순간 ID 가
        뒤바뀔 확률이 줄어든다**. 안 겹치는 쌍은 GIoU 가 음수라 자연히 걸러진다.
        """
        return giou_matrix(track_boxes, det_boxes)

    def _compensate(self, frame: Optional[np.ndarray]) -> None:
        if not self.use_cmc or frame is None:
            return
        import cv2  # CMC 를 쓸 때만 필요 (모듈 자체는 cv2 없이도 import 된다)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None or prev.shape != gray.shape or not self.tracks:
            return
        try:
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=self.cmc_max_corners, qualityLevel=0.01, minDistance=8)
            if p0 is None or len(p0) < 8:
                return
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
            if p1 is None or st is None:
                return
            ok = st.ravel() == 1
            if int(ok.sum()) < 8:
                return
            M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC)
            if M is None:
                return
            for t in self.tracks:
                t.kf.apply_affine(M)
        except Exception as e:  # 보정 실패가 추적 자체를 막으면 안 된다
            logger.warning(f"CMC 실패(무시하고 계속): {e}")


TRACKERS = {"sort": SortTracker, "bytetrack": ByteTrackTracker, "botsort": BotSortTracker}


def create_tracker(name: str = "botsort", **kwargs) -> ObjectTracker:
    """이름으로 트래커 생성 (camera_factory 와 같은 방식). 알 수 없는 이름이면 ValueError."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in TRACKERS:
        raise ValueError(f"알 수 없는 트래커: {name} (사용 가능: {', '.join(TRACKERS)})")
    return TRACKERS[key](**kwargs)
