"""
표면 추적 선 검출 알고리즘 (순수 모듈 — Qt 무의존)
====================================================
surface_tracking_tab.py 의 UI 와 동거하던 모듈 레벨 알고리즘을 분리했다
(opening_analysis.py / cad_registration.py 와 같은 원칙 — 앱 없이 단독 튜닝 가능).

  - detect_line               : 검은 선(적응 임계) / 컬러 마커(HSV) → 스켈레톤 마스크
  - trace_path_on_skeleton    : 스켈레톤 위 두 끝점 사이 경로 추적 (BFS)
  - sample_path_by_3d_distance: 경로를 3D 거리 간격으로 웨이포인트 샘플링
  - COLOR_HSV_RANGES          : 컬러 마커 HSV 기본 범위

알고리즘 상세는 docs/surface_tracking.md.
"""

import logging
from collections import deque
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# 컬러 마커 HSV 기본 범위 (OpenCV: H 0–179, S/V 0–255)
# 빨강은 H 둘레가 0/180에서 wrap-around 되므로 두 구간 OR.
# S/V 최저값은 default 70 — 사용자가 spin 으로 더 낮춰서 흐릿한 마커도 잡을 수 있음.
COLOR_HSV_RANGES = {
    "red": [((0, 70, 70), (10, 255, 255)), ((170, 70, 70), (180, 255, 255))],
    "blue": [((100, 70, 70), (130, 255, 255))],
    "green": [((40, 70, 70), (80, 255, 255))],
    "yellow": [((20, 70, 70), (35, 255, 255))],
    "magenta": [((140, 70, 70), (170, 255, 255))],
    "cyan": [((80, 70, 70), (100, 255, 255))],
}


def detect_line(
    bgr: np.ndarray,
    color_mode: str = "black",
    block_size: int = 21,
    threshold_c: int = 10,
    s_min: int = 70,
    v_min: int = 70,
    morph_kernel: int = 3,
    min_area: int = 200,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    마커 선 마스크 + 1픽셀 두께 skeleton 반환.

    `color_mode`:
        "black"  : adaptive threshold (어두운 매직 선)
        "red"/"blue"/"green"/"yellow"/"magenta"/"cyan" : HSV inRange 색 매칭

    Args:
        block_size, threshold_c : black 모드 전용
        s_min, v_min            : 컬러 모드 전용 (채도/명도 최저값)
        morph_kernel, min_area  : 공통 후처리
        roi : (x1, y1, x2, y2) — 이 영역 밖은 검출에서 제외

    Returns:
        (mask, skeleton) — 둘 다 uint8 (0/255), 검출 실패 시 (None, None)
    """
    if bgr is None:
        return None, None

    if color_mode == "black":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bs = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs,
            threshold_c,
        )
    else:
        ranges = COLOR_HSV_RANGES.get(color_mode)
        if ranges is None:
            logger.warning(f"알 수 없는 color_mode: {color_mode}")
            return None, None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        binary = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            # 사용자가 지정한 s_min/v_min 으로 채도·명도 임계 override
            lo = np.array([low[0], max(s_min, low[1]), max(v_min, low[2])], dtype=np.uint8)
            hi = np.array([high[0], high[1], high[2]], dtype=np.uint8)
            binary = cv2.bitwise_or(binary, cv2.inRange(hsv, lo, hi))

    k = max(1, morph_kernel)
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ROI 마스킹: 영역 밖 픽셀은 검출에서 배제 (배경 잡음을 매직 선으로 오인 방지)
    if roi is not None:
        h, w = binary.shape
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None, None
        roi_mask = np.zeros_like(binary)
        roi_mask[y1:y2, x1:x2] = 255
        binary = cv2.bitwise_and(binary, roi_mask)

    # 가장 큰 연결 성분만 유지 (배경 제외)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    if areas[largest - 1] < min_area:
        return None, None
    mask = ((labels == largest).astype(np.uint8)) * 255

    try:
        skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception as e:
        logger.warning(f"thinning 실패 → mask를 반환: {e}")
        skeleton = mask

    return mask, skeleton


def trace_path_on_skeleton(
    skeleton: np.ndarray,
    start_px: Tuple[int, int],
    end_px: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    skeleton(uint8 0/255)에서 start_px → end_px BFS 경로.
    실패 시 None. 클릭한 픽셀이 skeleton 위가 아니더라도 가장 가까운
    skeleton 픽셀에서 시작/끝으로 잡아준다.

    Returns: [(x, y), ...] (이미지 좌표)
    """
    pts = np.argwhere(skeleton > 0)  # (N, 2) (y, x)
    if len(pts) == 0:
        return None

    def nearest_yx(px):
        x, y = px
        dy = pts[:, 0] - y
        dx = pts[:, 1] - x
        d2 = dy * dy + dx * dx
        return tuple(pts[int(np.argmin(d2))])

    sy, sx = nearest_yx(start_px)
    ey, ex = nearest_yx(end_px)

    h, w = skeleton.shape
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sy, sx): None}
    q = deque([(sy, sx)])
    found = False
    while q:
        y, x = q.popleft()
        if (y, x) == (ey, ex):
            found = True
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_] > 0:
                    if (ny, nx_) not in parent:
                        parent[(ny, nx_)] = (y, x)
                        q.append((ny, nx_))

    if not found:
        return None

    path = []
    cur: Optional[Tuple[int, int]] = (ey, ex)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return [(x, y) for (y, x) in path]


def sample_path_by_3d_distance(
    path_pixels: List[Tuple[int, int]],
    xyz: np.ndarray,
    sampling_mm: float,
) -> List[int]:
    """
    path_pixels에 대응하는 3D 점들의 누적 거리를 따라 sampling_mm 간격으로 인덱스 선택.

    Returns:
        path_pixels 인덱스 리스트 (오름차순, 첫/끝점 포함)
    """
    valid_idx = []
    valid_pts = []
    for i, (px, py) in enumerate(path_pixels):
        p = xyz[py, px]
        if not np.any(np.isnan(p)):
            valid_idx.append(i)
            valid_pts.append(p)
    if len(valid_pts) < 2:
        return []

    valid_pts = np.array(valid_pts)
    diffs = np.diff(valid_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-3:
        return [valid_idx[0]]

    n_samples = max(2, int(np.floor(total / max(sampling_mm, 0.1))) + 1)
    targets = np.linspace(0.0, total, n_samples)

    selected: List[int] = []
    last_path_idx = -1
    for t in targets:
        k = int(np.argmin(np.abs(cum - t)))
        path_idx = valid_idx[k]
        if path_idx != last_path_idx:
            selected.append(path_idx)
            last_path_idx = path_idx
    return selected


# ============================================================
# Surface Tracking 탭
# ============================================================
