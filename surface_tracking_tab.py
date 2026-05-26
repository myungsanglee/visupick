"""
Surface Tracking 탭

자동차 외관처럼 굴곡진 표면 위에 매직펜으로 그린 검은 선을 카메라로 인식,
경로를 따라 로봇 툴이 표면에 수직(법선 방향)으로 정렬한 채 시작점→끝점까지
이동하는 모션을 생성한다.

파이프라인:
  1) 캡처 (RGB + XYZ + normals)
  2) 검은 선 검출 (adaptive threshold + morphology + thinning)
  3) 사용자가 2D 뷰에서 시작점/끝점 두 번 클릭
  4) skeleton 위에서 BFS path tracing
  5) 누적 3D 거리 기준 mm 간격 sampling
  6) 각 sample 의 법선(국소 평면 피팅) → Tool +Z = -normal 자세 계산
  7) offset mm 만큼 법선 바깥으로 띄움 (사용자 spin)
  8) 카메라 좌표계 → 로봇 base 좌표계
  9) QTimer 폴링으로 KRL 20슬롯 큐 동적 채움
"""

import logging
import json
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QPoint, QRect, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QStackedWidget,
    QApplication,
    QComboBox,
    QSplitter,
    QScrollArea,
)

import pyvista as pv

from calibration import tcp_to_homogeneous, compute_approach_pose, estimate_normal_at_pixel
from kuka_robot import normalize_robot_mode, is_auto_mode
from robot_control_mixin import RobotControlMixin
from bin_picking_tab import PointCloudView3D  # 3D 뷰 재사용
from image_view import ZoomableImageLabel

logger = logging.getLogger(__name__)


# ============================================================
# 2D 이미지 라벨: 시작점/끝점 두 번 클릭 모드
# ============================================================


class ClickPointImageLabel(ZoomableImageLabel):
    """
    검출 결과 오버레이 + 좌클릭 두 가지 동작 (베이스가 줌/팬 처리):
      - 짧은 클릭 (≤ CLICK_THRESHOLD): 시작점/끝점 지정
      - 드래그 (> CLICK_THRESHOLD): ROI 사각형 설정
    """

    pointClicked = Signal(int, int)
    roiChanged = Signal(int, int, int, int)

    CLICK_THRESHOLD = 8

    def __init__(self):
        super().__init__()
        self._overlay_bgr: Optional[np.ndarray] = None
        self._start_pt: Optional[Tuple[int, int]] = None
        self._end_pt: Optional[Tuple[int, int]] = None
        self._path_pixels: Optional[List[Tuple[int, int]]] = None
        self._sample_pixels: Optional[List[Tuple[int, int]]] = None
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None

        # 좌클릭 드래그 상태
        self._dragging = False
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None

    def set_overlay(self, overlay_bgr: Optional[np.ndarray]):
        self._overlay_bgr = overlay_bgr
        self._refresh()

    def set_endpoints(self, start: Optional[Tuple[int, int]], end: Optional[Tuple[int, int]]):
        self._start_pt = start
        self._end_pt = end
        self._refresh()

    def set_path(self, path_pixels: Optional[List[Tuple[int, int]]]):
        self._path_pixels = path_pixels
        self._refresh()

    def set_samples(self, sample_pixels: Optional[List[Tuple[int, int]]]):
        self._sample_pixels = sample_pixels
        self._refresh()

    def set_roi(self, rect: Optional[Tuple[int, int, int, int]]):
        self._roi_rect = rect
        self._refresh()

    def clear_all(self):
        self._overlay_bgr = None
        self._start_pt = None
        self._end_pt = None
        self._path_pixels = None
        self._sample_pixels = None
        self._roi_rect = None
        self.clear_image()

    # 이미지 좌표계 오버레이 (skeleton, path, samples, endpoints, ROI)
    def _make_overlay_image(self) -> Optional[np.ndarray]:
        if self._bgr is None:
            return None
        canvas = self._bgr.copy()
        if self._overlay_bgr is not None and self._overlay_bgr.shape == canvas.shape:
            canvas = cv2.addWeighted(canvas, 0.55, self._overlay_bgr, 0.45, 0)

        if self._path_pixels and len(self._path_pixels) >= 2:
            pts = np.array(self._path_pixels, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

        if self._sample_pixels:
            for i, (sx, sy) in enumerate(self._sample_pixels):
                cv2.circle(canvas, (int(sx), int(sy)), 4, (0, 165, 255), -1)
                cv2.putText(canvas, str(i + 1), (int(sx) + 5, int(sy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        if self._start_pt is not None:
            cv2.circle(canvas, self._start_pt, 8, (0, 255, 0), 2)
            cv2.putText(canvas, "S", (self._start_pt[0] + 10, self._start_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self._end_pt is not None:
            cv2.circle(canvas, self._end_pt, 8, (0, 0, 255), 2)
            cv2.putText(canvas, "E", (self._end_pt[0] + 10, self._end_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self._roi_rect is not None:
            rx1, ry1, rx2, ry2 = self._roi_rect
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(canvas, "ROI", (rx1, max(ry1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return canvas

    # 위젯 좌표계 임시 드래그 사각형
    def _post_draw(self, pixmap):
        if not (self._dragging and self._drag_start and self._drag_current):
            return
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())
        painter.end()

    # 좌클릭만 처리, 우클릭은 베이스 (팬) 에 위임
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._bgr is not None:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_current = event.position().toPoint()
            self._refresh()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._dragging:
            super().mouseReleaseEvent(event)
            return

        self._dragging = False
        end = event.position().toPoint()
        start = self._drag_start
        self._drag_start = None
        self._drag_current = None
        if start is None:
            self._refresh()
            return

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        dist = (dx * dx + dy * dy) ** 0.5

        if dist <= self.CLICK_THRESHOLD:
            pt = self._widget_to_image(end)
            if pt is not None:
                self.pointClicked.emit(pt[0], pt[1])
            self._refresh()
            return

        s_img = self._widget_to_image(start)
        e_img = self._widget_to_image(end)
        if s_img and e_img:
            x1, y1 = s_img
            x2, y2 = e_img
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self._roi_rect = (x1, y1, x2, y2)
                self.roiChanged.emit(x1, y1, x2, y2)
        self._refresh()


# ============================================================
# 검은 선 검출 + path tracing
# ============================================================


# 컬러 마커 HSV 기본 범위 (OpenCV: H 0–179, S/V 0–255)
# 빨강은 H 둘레가 0/180에서 wrap-around 되므로 두 구간 OR.
# S/V 최저값은 default 70 — 사용자가 spin 으로 더 낮춰서 흐릿한 마커도 잡을 수 있음.
COLOR_HSV_RANGES = {
    "red":     [((0, 70, 70), (10, 255, 255)),
                ((170, 70, 70), (180, 255, 255))],
    "blue":    [((100, 70, 70), (130, 255, 255))],
    "green":   [((40, 70, 70), (80, 255, 255))],
    "yellow":  [((20, 70, 70), (35, 255, 255))],
    "magenta": [((140, 70, 70), (170, 255, 255))],
    "cyan":    [((80, 70, 70), (100, 255, 255))],
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
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs, threshold_c,
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


class SurfaceTrackingTab(RobotControlMixin, QWidget):
    """
    매직펜으로 그린 검은 선을 따라 로봇 툴이 표면 법선 방향으로 정렬한 채
    이동하는 시나리오를 생성/실행.
    """

    SEQ_OBJECT_NOUN = "경로점"

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        # 캡처 데이터
        self.current_image = None       # BGR
        self.current_xyz = None         # (H, W, 3) mm
        self.current_normals = None     # (H, W, 3) 또는 None
        self.current_intrinsics = None
        self.current_rgb = None

        # 검출 결과
        self.line_mask = None
        self.skeleton = None

        # 클릭한 끝점
        self.start_pt: Optional[Tuple[int, int]] = None
        self.end_pt: Optional[Tuple[int, int]] = None

        # ROI (선택적): 이 영역 안에서만 검은 선 검출
        self.roi_2d: Optional[Tuple[int, int, int, int]] = None

        # 경로
        self.path_pixels: Optional[List[Tuple[int, int]]] = None
        self.sample_indices: List[int] = []
        self.path_points: List[Dict] = []   # 각 sample 에 대해 robot base 좌표계 TCP 자세

        # 캘리브레이션
        self.T_calib = None
        self.calib_mode = None

        # 실행 상태
        self._motion_running = False
        self._pending_points: List[Dict] = []   # 아직 KRL 큐에 보내지 못한 점
        self._sent_count = 0
        self._total_count = 0
        self._send_timer: Optional[QTimer] = None
        self._completion_timer: Optional[QTimer] = None

        # RobotControlMixin이 참조하는 속성 (시퀀스 큐 기능은 안 쓰지만 mixin 호환을 위해)
        self.user_queue: List[Dict] = []
        self.selected_idx = None
        self.target_pose = None
        self._tcp_viz_actors: List[str] = []

        self._current_mode = "?"

        self._init_ui()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 상단 컨트롤
        top_row = QHBoxLayout()

        self.btn_load_calib = QPushButton("캘리브레이션 (JSON)")
        self.btn_load_calib.clicked.connect(self._load_calibration)
        top_row.addWidget(self.btn_load_calib)
        self.calib_label = QLabel("미로드")
        top_row.addWidget(self.calib_label)
        top_row.addSpacing(15)

        self.btn_capture = QPushButton("캡처")
        self.btn_capture.clicked.connect(self._capture)
        top_row.addWidget(self.btn_capture)

        self.btn_clear_roi = QPushButton("ROI 해제")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        self.btn_clear_roi.setToolTip("이미지에서 드래그로 그린 검출 ROI 사각형을 해제")
        top_row.addWidget(self.btn_clear_roi)

        self.btn_detect = QPushButton("선 검출")
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_detect.clicked.connect(self._detect_line)
        top_row.addWidget(self.btn_detect)

        self.btn_clear_pts = QPushButton("끝점 해제")
        self.btn_clear_pts.clicked.connect(self._clear_endpoints)
        top_row.addWidget(self.btn_clear_pts)

        self.btn_compute_path = QPushButton("경로 계산")
        self.btn_compute_path.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.btn_compute_path.clicked.connect(self._compute_path)
        self.btn_compute_path.setEnabled(False)
        top_row.addWidget(self.btn_compute_path)

        top_row.addStretch()

        self.mode_label = QLabel("모드: ?")
        self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; "
                                      "background-color: #BDBDBD; color: white; border-radius: 3px;")
        top_row.addWidget(self.mode_label)

        self.btn_view_2d = QPushButton("2D 뷰")
        self.btn_view_2d.setCheckable(True)
        self.btn_view_2d.setChecked(True)
        self.btn_view_2d.clicked.connect(lambda: self._switch_view(0))
        top_row.addWidget(self.btn_view_2d)

        self.btn_view_3d = QPushButton("3D 뷰")
        self.btn_view_3d.setCheckable(True)
        self.btn_view_3d.clicked.connect(lambda: self._switch_view(1))
        top_row.addWidget(self.btn_view_3d)

        top_widget = QWidget()
        top_widget.setLayout(top_row)
        top_widget.setFixedHeight(top_widget.sizeHint().height())
        layout.addWidget(top_widget)

        # 중앙 splitter
        splitter = QSplitter(Qt.Horizontal)

        self.view_stack = QStackedWidget()
        self.view_2d = ClickPointImageLabel()
        self.view_2d.pointClicked.connect(self._on_image_click)
        self.view_2d.roiChanged.connect(self._on_roi_dragged)
        self.view_stack.addWidget(self.view_2d)

        self.view_3d = PointCloudView3D()
        self.view_stack.addWidget(self.view_3d)

        splitter.addWidget(self.view_stack)

        # 우측 정보 패널
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # 검출 파라미터
        det_group = QGroupBox("선 검출 파라미터")
        det_layout = QVBoxLayout(det_group)

        # 마커 색 콤보 (검정 / 컬러)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("마커 색:"))
        self.color_combo = QComboBox()
        # 표시 라벨 → 내부 키
        self._color_options = [
            ("검정 (adaptive threshold)", "black"),
            ("빨강", "red"),
            ("파랑", "blue"),
            ("초록", "green"),
            ("노랑", "yellow"),
            ("자홍 (magenta)", "magenta"),
            ("청록 (cyan)", "cyan"),
        ]
        for label, _ in self._color_options:
            self.color_combo.addItem(label)
        self.color_combo.setCurrentIndex(1)  # 기본 빨강
        self.color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        color_row.addWidget(self.color_combo)
        color_row.addStretch()
        det_layout.addLayout(color_row)

        # 검정 모드용 파라미터
        bs_row = QHBoxLayout()
        bs_row.addWidget(QLabel("Block size:"))
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 99)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(21)
        self.block_size_spin.setToolTip("(검정 모드) Adaptive threshold 블록 크기 (홀수)")
        bs_row.addWidget(self.block_size_spin)
        bs_row.addWidget(QLabel("C:"))
        self.thresh_c_spin = QSpinBox()
        self.thresh_c_spin.setRange(-20, 50)
        self.thresh_c_spin.setValue(10)
        self.thresh_c_spin.setToolTip("(검정 모드) Threshold 보정 상수")
        bs_row.addWidget(self.thresh_c_spin)
        bs_row.addStretch()
        det_layout.addLayout(bs_row)

        # 컬러 모드용 파라미터
        sv_row = QHBoxLayout()
        sv_row.addWidget(QLabel("S min:"))
        self.s_min_spin = QSpinBox()
        self.s_min_spin.setRange(0, 255)
        self.s_min_spin.setValue(70)
        self.s_min_spin.setToolTip("(컬러 모드) 채도 최저값 — 낮을수록 흐릿한 색도 잡음")
        sv_row.addWidget(self.s_min_spin)
        sv_row.addWidget(QLabel("V min:"))
        self.v_min_spin = QSpinBox()
        self.v_min_spin.setRange(0, 255)
        self.v_min_spin.setValue(70)
        self.v_min_spin.setToolTip("(컬러 모드) 명도 최저값 — 어두운 음영도 잡으려면 줄임")
        sv_row.addWidget(self.v_min_spin)
        sv_row.addStretch()
        det_layout.addLayout(sv_row)

        morph_row = QHBoxLayout()
        morph_row.addWidget(QLabel("Morph kernel:"))
        self.morph_spin = QSpinBox()
        self.morph_spin.setRange(1, 9)
        self.morph_spin.setValue(3)
        self.morph_spin.setToolTip("노이즈 제거 + 선 연결용 morphology 커널 크기 (공통)")
        morph_row.addWidget(self.morph_spin)
        morph_row.addStretch()
        det_layout.addLayout(morph_row)

        info_layout.addWidget(det_group)
        # 시작 시 모드별 파라미터 활성화 토글
        self._on_color_mode_changed(self.color_combo.currentIndex())

        # 끝점 표시
        pt_group = QGroupBox("끝점 (이미지에서 클릭)")
        pt_layout = QVBoxLayout(pt_group)
        self.start_label = QLabel("시작점: 미설정")
        self.end_label = QLabel("끝점: 미설정")
        pt_layout.addWidget(self.start_label)
        pt_layout.addWidget(self.end_label)
        self.roi_label = QLabel("ROI: 전체 이미지")
        self.roi_label.setStyleSheet("color: #888;")
        pt_layout.addWidget(self.roi_label)
        pt_layout.addWidget(QLabel("· 짧게 클릭 → 시작점/끝점 설정\n· 드래그 → 검출 ROI 사각형"))
        info_layout.addWidget(pt_group)

        # 경로 sampling 파라미터
        path_group = QGroupBox("경로 sampling")
        path_layout = QVBoxLayout(path_group)

        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("간격 (mm):"))
        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(1.0, 200.0)
        self.sampling_spin.setSingleStep(1.0)
        self.sampling_spin.setValue(10.0)
        self.sampling_spin.setDecimals(1)
        self.sampling_spin.setToolTip("이웃 경로점 사이의 3D 누적 거리 (mm)")
        s_row.addWidget(self.sampling_spin)
        s_row.addStretch()
        path_layout.addLayout(s_row)

        o_row = QHBoxLayout()
        o_row.addWidget(QLabel("Offset (mm):"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-200.0, 200.0)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setValue(20.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setToolTip("표면 법선 바깥 방향으로 TCP가 떨어진 거리 — 0 이면 표면 접촉")
        o_row.addWidget(self.offset_spin)
        o_row.addStretch()
        path_layout.addLayout(o_row)

        np_row = QHBoxLayout()
        np_row.addWidget(QLabel("법선 패치 반경(px):"))
        self.normal_patch_spin = QSpinBox()
        self.normal_patch_spin.setRange(3, 60)
        self.normal_patch_spin.setValue(12)
        self.normal_patch_spin.setToolTip("각 sample 의 법선 추정을 위한 국소 평면 패치 반경")
        np_row.addWidget(self.normal_patch_spin)
        np_row.addStretch()
        path_layout.addLayout(np_row)

        self.path_info_label = QLabel("경로점: 0개")
        self.path_info_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        path_layout.addWidget(self.path_info_label)

        info_layout.addWidget(path_group)

        # 실행 제어
        exec_group = QGroupBox("실행 제어")
        exec_layout = QVBoxLayout(exec_group)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("속도(%):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(20)
        self.speed_spin.setFixedWidth(70)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_spin)
        self.btn_apply_speed = QPushButton("적용")
        self.btn_apply_speed.setFixedWidth(50)
        self.btn_apply_speed.clicked.connect(self._apply_speed_now)
        speed_row.addWidget(self.btn_apply_speed)
        speed_row.addStretch()
        exec_layout.addLayout(speed_row)

        zlim_row = QHBoxLayout()
        zlim_row.addWidget(QLabel("Z 최소(mm):"))
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(-2000, 2000)
        self.z_min_spin.setValue(5)
        self.z_min_spin.setFixedWidth(80)
        self.z_min_spin.setToolTip("경로 어느 점이라도 이 값보다 낮으면 실행을 거부")
        zlim_row.addWidget(self.z_min_spin)
        zlim_row.addStretch()
        exec_layout.addLayout(zlim_row)

        # 이동 방식 (RobotControlMixin이 참조하지만 여기서는 항상 LIN)
        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(["LIN (직선, 고정)"])
        self.move_mode_combo.setEnabled(False)

        # 진행 상황
        self.progress_label = QLabel("대기 중")
        self.progress_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        exec_layout.addWidget(self.progress_label)

        # 시작 / 정지
        self.btn_start = QPushButton("▶ 경로 추적 시작")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; "
                                     "background-color: #1565C0; color: white;")
        self.btn_start.clicked.connect(self._start_path_motion)
        self.btn_start.setEnabled(False)
        exec_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■ 정지 (남은 큐 비우기)")
        self.btn_stop.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self._stop_path_motion)
        self.btn_stop.setEnabled(False)
        exec_layout.addWidget(self.btn_stop)

        # Home (RobotControlMixin)
        home_row = QHBoxLayout()
        self.btn_move_home = QPushButton("🏠 Home으로 이동")
        self.btn_move_home.setMinimumHeight(40)
        self.btn_move_home.setStyleSheet("font-size: 13px; font-weight: bold; "
                                         "background-color: #2E7D32; color: white;")
        self.btn_move_home.clicked.connect(self._move_to_home)
        self.btn_move_home.setEnabled(False)
        home_row.addWidget(self.btn_move_home, stretch=2)

        self.btn_set_home = QPushButton("📍 Home\n재설정")
        self.btn_set_home.setMinimumHeight(40)
        self.btn_set_home.setStyleSheet("font-size: 11px; background-color: #689F38; color: white;")
        self.btn_set_home.clicked.connect(self._set_home_to_current)
        self.btn_set_home.setEnabled(False)
        home_row.addWidget(self.btn_set_home, stretch=1)
        exec_layout.addLayout(home_row)

        self.btn_clear_queue = QPushButton("🗑 큐 비우기")
        self.btn_clear_queue.setStyleSheet("background-color: #F57C00; color: white;")
        self.btn_clear_queue.clicked.connect(self._clear_motion_queue)
        exec_layout.addWidget(self.btn_clear_queue)

        self.btn_estop = QPushButton("⛔ 비상정지 (Space)")
        self.btn_estop.setMinimumHeight(60)
        self.btn_estop.setStyleSheet("font-size: 16px; font-weight: bold; "
                                     "background-color: #D32F2F; color: white;")
        self.btn_estop.clicked.connect(self._emergency_stop)
        exec_layout.addWidget(self.btn_estop)

        self.btn_estop_release = QPushButton("비상정지 해제")
        self.btn_estop_release.setStyleSheet("background-color: #757575; color: white;")
        self.btn_estop_release.clicked.connect(self._emergency_stop_release)
        exec_layout.addWidget(self.btn_estop_release)

        info_layout.addWidget(exec_group)
        info_layout.addStretch()

        # Mixin이 참조하지만 SurfaceTracking 에서는 안 쓰는 위젯들 (호환용 hidden)
        self.use_approach = self._make_hidden_checkbox(True)
        self.approach_dist = self._make_hidden_spinbox(50)
        self.btn_move = self.btn_start  # alias
        self.btn_add_obj_to_seq = self._make_hidden_button()
        self.btn_add_home_to_seq = self._make_hidden_button()
        from PySide6.QtWidgets import QListWidget
        self.action_list = QListWidget()

        info_scroll = QScrollArea()
        info_scroll.setWidget(info_widget)
        info_scroll.setWidgetResizable(True)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setMinimumWidth(380)

        splitter.addWidget(info_scroll)
        splitter.setSizes([850, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # Space = 비상정지
        sc_estop = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_estop.setContext(Qt.WidgetWithChildrenShortcut)
        sc_estop.activated.connect(self._emergency_stop)

        # 모드 표시 주기 갱신
        self._mode_timer = QTimer(self)
        self._mode_timer.timeout.connect(self._refresh_mode_display)
        self._mode_timer.start(2000)

    def _make_hidden_checkbox(self, checked: bool):
        from PySide6.QtWidgets import QCheckBox
        cb = QCheckBox()
        cb.setChecked(checked)
        cb.setVisible(False)
        return cb

    def _make_hidden_spinbox(self, value: int):
        sb = QSpinBox()
        sb.setRange(-9999, 9999)
        sb.setValue(value)
        sb.setVisible(False)
        return sb

    def _make_hidden_button(self):
        btn = QPushButton()
        btn.setVisible(False)
        return btn

    def _switch_view(self, idx: int):
        self.view_stack.setCurrentIndex(idx)
        self.btn_view_2d.setChecked(idx == 0)
        self.btn_view_3d.setChecked(idx == 1)
        if idx == 1:
            QTimer.singleShot(0, self.view_3d.refresh_camera)

    # ------------------------------------------------------------
    # 캘리브레이션 / 캡처
    # ------------------------------------------------------------

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "캘리브레이션 파일 선택", "data", "JSON Files (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            with open(path) as f:
                result = json.load(f)
            self.T_calib = np.array(result["transformation_matrix"])
            self.calib_mode = result.get("mode", "eye_to_hand")
            self.calib_label.setText(f"{Path(path).name} [{self.calib_mode}]")
            self.main.statusBar().showMessage(f"캘리브레이션 로드: {self.calib_mode}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"로드 실패:\n{e}")

    def _capture(self):
        if not self.main.camera or not self.main.camera.connected:
            QMessageBox.warning(self, "오류", "카메라가 연결되지 않았습니다")
            return
        if not self.main.camera.is_capture_ready:
            QMessageBox.warning(self, "오류", "카메라가 캡처 준비되지 않았습니다 (Zivid 는 YML 로드 필요)")
            return

        self.main.statusBar().showMessage("캡처 중...")
        QApplication.processEvents()

        frame = self.main.camera.capture()
        if frame is None:
            self.main.statusBar().showMessage("캡처 실패")
            return

        image = self.main.camera.frame_to_2d_image(frame)
        xyz = self.main.camera.frame_to_point_cloud(frame)
        normals = self.main.camera.frame_to_normals(frame)
        if image is None or xyz is None:
            self.main.statusBar().showMessage("데이터 추출 실패")
            return

        self.current_image = image
        self.current_xyz = xyz
        self.current_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_normals = normals

        intr_data = self.main.camera.get_intrinsics()
        if intr_data:
            self.current_intrinsics = np.array(intr_data["camera_matrix"])

        # 상태 리셋 (ROI는 사용자가 원하면 캡처 후에도 재사용 가능하니 유지)
        self.line_mask = None
        self.skeleton = None
        self.start_pt = None
        self.end_pt = None
        self.path_pixels = None
        self.sample_indices = []
        self.path_points = []
        self.btn_compute_path.setEnabled(False)
        self.btn_start.setEnabled(False)
        self._update_endpoint_labels()
        self._update_path_info()

        # 2D 뷰 갱신 (ROI는 유지)
        self.view_2d.set_image(image)
        self.view_2d.set_overlay(None)
        self.view_2d.set_endpoints(None, None)
        self.view_2d.set_path(None)
        self.view_2d.set_samples(None)
        self.view_2d.set_roi(self.roi_2d)

        # 3D 뷰 갱신
        self.view_3d.clear()
        self.view_3d.show_pointcloud(
            xyz, self.current_rgb,
            intrinsics=self.current_intrinsics,
            image_shape=image.shape,
        )
        self.view_3d.reset_view()

        self.main.statusBar().showMessage("캡처 완료 — '검은 선 검출' 버튼")

    # ------------------------------------------------------------
    # 검은 선 검출
    # ------------------------------------------------------------

    def _current_color_mode(self) -> str:
        idx = self.color_combo.currentIndex()
        return self._color_options[idx][1]

    def _on_color_mode_changed(self, idx: int):
        """검정/컬러 모드 전환 시 각 모드의 파라미터 spin 활성화 토글."""
        mode = self._color_options[idx][1] if 0 <= idx < len(self._color_options) else "black"
        is_black = (mode == "black")
        self.block_size_spin.setEnabled(is_black)
        self.thresh_c_spin.setEnabled(is_black)
        self.s_min_spin.setEnabled(not is_black)
        self.v_min_spin.setEnabled(not is_black)

    def _detect_line(self):
        if self.current_image is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return
        color_mode = self._current_color_mode()
        color_label = self.color_combo.currentText().split(" ")[0]
        self.main.statusBar().showMessage(f"{color_label} 선 검출 중...")
        QApplication.processEvents()

        mask, skel = detect_line(
            self.current_image,
            color_mode=color_mode,
            block_size=self.block_size_spin.value(),
            threshold_c=self.thresh_c_spin.value(),
            s_min=self.s_min_spin.value(),
            v_min=self.v_min_spin.value(),
            morph_kernel=self.morph_spin.value(),
            roi=self.roi_2d,
        )
        if mask is None:
            scope = "ROI 영역에서 " if self.roi_2d is not None else ""
            QMessageBox.warning(
                self, "검출 실패",
                f"{scope}{color_label} 선을 찾지 못했습니다. 파라미터/ROI를 조정해 보세요."
            )
            self.line_mask = None
            self.skeleton = None
            self.view_2d.set_overlay(None)
            return

        self.line_mask = mask
        self.skeleton = skel

        # 시각화 오버레이: mask 회색 + skeleton 시안 (빨강 마커와 색 겹치지 않게)
        overlay = np.zeros_like(self.current_image)
        overlay[mask > 0] = (60, 60, 60)
        overlay[skel > 0] = (255, 255, 0)   # BGR=노랑·시안 계열 (실제로는 시안)
        self.view_2d.set_overlay(overlay)

        n_skel = int(np.count_nonzero(skel))
        n_mask = int(np.count_nonzero(mask))
        scope_msg = " (ROI 영역)" if self.roi_2d is not None else ""
        self.main.statusBar().showMessage(
            f"{color_label} 선 검출 완료{scope_msg}: mask {n_mask} px, "
            f"skeleton {n_skel} px — 시작점/끝점 클릭"
        )

    # ------------------------------------------------------------
    # 끝점 클릭 / 경로 계산
    # ------------------------------------------------------------

    def _on_image_click(self, x: int, y: int):
        if self.skeleton is None:
            QMessageBox.information(self, "안내", "먼저 '검은 선 검출' 을 누르세요")
            return

        # ROI가 설정되어 있으면 그 안에서만 끝점 받음 (밖이면 무시)
        if self.roi_2d is not None:
            rx1, ry1, rx2, ry2 = self.roi_2d
            if not (rx1 <= x <= rx2 and ry1 <= y <= ry2):
                self.main.statusBar().showMessage("ROI 밖 클릭 — 무시됨")
                return

        # 첫 클릭 = 시작, 둘째 클릭 = 끝, 셋째 클릭 = 다시 시작부터
        if self.start_pt is None or (self.start_pt is not None and self.end_pt is not None):
            self.start_pt = (x, y)
            self.end_pt = None
            self.path_pixels = None
            self.sample_indices = []
            self.view_2d.set_path(None)
            self.view_2d.set_samples(None)
        else:
            self.end_pt = (x, y)

        self.view_2d.set_endpoints(self.start_pt, self.end_pt)
        self._update_endpoint_labels()
        self.btn_compute_path.setEnabled(self.start_pt is not None and self.end_pt is not None)

    def _on_roi_dragged(self, x1: int, y1: int, x2: int, y2: int):
        """드래그로 ROI 사각형이 새로 설정됨 → 검출 캐시 무효화."""
        self.roi_2d = (x1, y1, x2, y2)
        # ROI가 바뀌면 이전 검출/경로는 영역과 안 맞을 수 있으니 정리
        self.line_mask = None
        self.skeleton = None
        self.path_pixels = None
        self.sample_indices = []
        self.path_points = []
        self.start_pt = None
        self.end_pt = None
        self.view_2d.set_overlay(None)
        self.view_2d.set_path(None)
        self.view_2d.set_samples(None)
        self.view_2d.set_endpoints(None, None)
        self._update_endpoint_labels()
        self._update_path_info()
        self._update_roi_label()
        self._clear_3d_path_markers()
        self.btn_compute_path.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.main.statusBar().showMessage(
            f"ROI 설정: ({x1},{y1})–({x2},{y2}) — 검출 다시 실행"
        )

    def _clear_roi(self):
        if self.roi_2d is None:
            return
        self.roi_2d = None
        self.view_2d.set_roi(None)
        # 이전 검출 결과도 ROI에 종속이므로 무효화
        self.line_mask = None
        self.skeleton = None
        self.path_pixels = None
        self.sample_indices = []
        self.path_points = []
        self.start_pt = None
        self.end_pt = None
        self.view_2d.set_overlay(None)
        self.view_2d.set_path(None)
        self.view_2d.set_samples(None)
        self.view_2d.set_endpoints(None, None)
        self._update_endpoint_labels()
        self._update_path_info()
        self._update_roi_label()
        self._clear_3d_path_markers()
        self.btn_compute_path.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.main.statusBar().showMessage("ROI 해제됨 (전체 이미지 검출)")

    def _update_roi_label(self):
        if self.roi_2d is None:
            self.roi_label.setText("ROI: 전체 이미지")
            self.roi_label.setStyleSheet("color: #888;")
        else:
            x1, y1, x2, y2 = self.roi_2d
            self.roi_label.setText(f"ROI: ({x1},{y1})–({x2},{y2})  {x2 - x1}×{y2 - y1}")
            self.roi_label.setStyleSheet("color: #0066cc; font-weight: bold;")

    def _clear_endpoints(self):
        self.start_pt = None
        self.end_pt = None
        self.path_pixels = None
        self.sample_indices = []
        self.path_points = []
        self.view_2d.set_endpoints(None, None)
        self.view_2d.set_path(None)
        self.view_2d.set_samples(None)
        self._clear_3d_path_markers()
        self._update_endpoint_labels()
        self._update_path_info()
        self.btn_compute_path.setEnabled(False)
        self.btn_start.setEnabled(False)

    def _update_endpoint_labels(self):
        self.start_label.setText(
            f"시작점: {self.start_pt}" if self.start_pt else "시작점: 미설정"
        )
        self.end_label.setText(
            f"끝점:   {self.end_pt}" if self.end_pt else "끝점: 미설정"
        )

    def _update_path_info(self):
        n = len(self.path_points)
        self.path_info_label.setText(f"경로점: {n}개")

    def _compute_path(self):
        if self.skeleton is None or self.start_pt is None or self.end_pt is None:
            return
        if self.T_calib is None:
            QMessageBox.warning(self, "오류", "캘리브레이션 파일을 먼저 로드하세요")
            return
        if self.current_xyz is None:
            return

        # 1) BFS path
        path_px = trace_path_on_skeleton(self.skeleton, self.start_pt, self.end_pt)
        if path_px is None or len(path_px) < 2:
            QMessageBox.warning(
                self, "경로 검색 실패",
                "시작점→끝점 사이에서 skeleton 위 경로를 찾지 못했습니다.\n"
                "검출 파라미터를 조정하거나 끝점 위치를 바꿔 보세요."
            )
            return
        self.path_pixels = path_px
        self.view_2d.set_path(path_px)

        # 2) 누적 3D 거리 sampling
        self.sample_indices = sample_path_by_3d_distance(
            path_px, self.current_xyz, self.sampling_spin.value()
        )
        if len(self.sample_indices) < 2:
            QMessageBox.warning(self, "오류", "유효한 3D 점이 부족해 sampling 실패")
            return

        sample_pixels = [path_px[i] for i in self.sample_indices]
        self.view_2d.set_samples(sample_pixels)

        # 3) 각 sample 의 카메라 좌표계 3D + 법선
        cam_points = []
        cam_normals = []
        patch_r = self.normal_patch_spin.value()
        for (px, py) in sample_pixels:
            p3d = self.current_xyz[py, px]
            if np.any(np.isnan(p3d)):
                continue
            n_cam = estimate_normal_at_pixel(
                self.current_xyz, px, py,
                patch_radius=patch_r,
                normals=self.current_normals,
            )
            if n_cam is None:
                continue
            cam_points.append(p3d)
            cam_normals.append(n_cam)

        if len(cam_points) < 2:
            QMessageBox.warning(self, "오류", "유효한 법선을 가진 sample 부족")
            return

        cam_points = np.array(cam_points)
        cam_normals = np.array(cam_normals)

        # 4) 카메라 → base 변환
        if self.calib_mode == "eye_to_hand":
            R_c2b = self.T_calib[:3, :3]
            base_points = (self.T_calib @ np.hstack(
                [cam_points, np.ones((len(cam_points), 1))]
            ).T).T[:, :3]
            base_normals = (R_c2b @ cam_normals.T).T
        elif self.calib_mode == "eye_in_hand":
            if self.main.robot is None:
                QMessageBox.warning(self, "오류", "Eye-in-Hand 모드는 로봇 연결 필요")
                return
            cur_tcp = self.main.robot.get_tcp_position()
            T_g2b = tcp_to_homogeneous(cur_tcp)
            T_c2b = T_g2b @ self.T_calib
            R_c2b = T_c2b[:3, :3]
            base_points = (T_c2b @ np.hstack(
                [cam_points, np.ones((len(cam_points), 1))]
            ).T).T[:, :3]
            base_normals = (R_c2b @ cam_normals.T).T
        else:
            QMessageBox.critical(self, "오류", f"알 수 없는 모드: {self.calib_mode}")
            return

        # 5) 각 점에서 Tool +Z = -normal 자세 + offset
        cur_tcp = None
        if self.main.robot is not None:
            cur_tcp = self.main.robot.get_tcp_position()
        if cur_tcp is None:
            cur_tcp = {"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 180}

        offset_mm = self.offset_spin.value()
        path_points: List[Dict] = []
        prev_tcp = cur_tcp
        for p_base, n_base in zip(base_points, base_normals):
            n_unit = n_base / (np.linalg.norm(n_base) + 1e-9)
            # offset 만큼 법선 바깥(표면에서 멀어지는 방향)으로 이동
            shifted = p_base + n_unit * offset_mm
            pose = compute_approach_pose(shifted, n_base, prev_tcp)
            path_points.append(pose)
            # ABC unwrap이 이전 자세에 가까운 표현을 선택하도록 prev를 갱신
            prev_tcp = pose

        self.path_points = path_points

        # 6) 3D 뷰에 경로 + 법선 시각화 (카메라 좌표계)
        self._render_3d_path(cam_points, cam_normals)

        self._update_path_info()
        self.btn_start.setEnabled(self.main.robot is not None and len(self.path_points) >= 2)
        self.main.statusBar().showMessage(
            f"경로 계산 완료: {len(self.path_points)}점 "
            f"(간격 {self.sampling_spin.value():.1f}mm, offset {offset_mm:+.1f}mm)"
        )

    # ------------------------------------------------------------
    # 3D 시각화
    # ------------------------------------------------------------

    def _clear_3d_path_markers(self):
        plotter = self.view_3d.plotter
        for name in self._tcp_viz_actors:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._tcp_viz_actors.clear()
        plotter.render()

    def _render_3d_path(self, cam_points: np.ndarray, cam_normals: np.ndarray):
        """카메라 좌표계에서 sample 점 + 법선 화살표 + 경로선 표시."""
        self._clear_3d_path_markers()
        plotter = self.view_3d.plotter

        # 경로 점 (작은 sphere 글리프)
        cloud = pv.PolyData(cam_points.astype(np.float32))
        sphere_glyph = cloud.glyph(geom=pv.Sphere(radius=3), scale=False, orient=False)
        plotter.add_mesh(
            sphere_glyph, color="orange",
            name="path_spheres", pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("path_spheres")

        # 경로 선 (연결)
        if len(cam_points) >= 2:
            n = len(cam_points)
            lines = np.hstack([[2, i, i + 1] for i in range(n - 1)]).astype(np.int32)
            poly = pv.PolyData(cam_points.astype(np.float32))
            poly.lines = lines
            plotter.add_mesh(
                poly, color="yellow", line_width=4,
                render_lines_as_tubes=True, name="path_line",
                pickable=False, reset_camera=False,
            )
            self._tcp_viz_actors.append("path_line")

        # 각 점의 법선 화살표 (카메라 좌표계 -normal = 표면 안쪽이지만,
        # 시각화는 표면 바깥쪽 +normal 으로 표시 — Tool +Z 는 -normal 방향)
        arrows = pv.PolyData(cam_points.astype(np.float32))
        arrows["vectors"] = (cam_normals * 30.0).astype(np.float32)
        arrow_glyph = arrows.glyph(geom=pv.Arrow(), orient="vectors", scale="vectors", factor=1.0)
        plotter.add_mesh(
            arrow_glyph, color="cyan",
            name="path_normals", pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("path_normals")

        plotter.render()

    # ------------------------------------------------------------
    # 실행 (동적 큐 채움)
    # ------------------------------------------------------------

    def _start_path_motion(self):
        if not self.path_points or len(self.path_points) < 2:
            QMessageBox.warning(self, "오류", "경로점이 없습니다 — 먼저 '경로 계산'")
            return
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        if self._motion_running:
            QMessageBox.information(self, "안내", "이미 실행 중입니다")
            return

        # 전 점 Z 검증
        for i, p in enumerate(self.path_points):
            if not self._validate_z(p["z"]):
                self.main.statusBar().showMessage(f"⛔ {i + 1}번 점 Z 한계 초과 — 실행 거부")
                return

        speed = self._effective_speed(self.speed_spin.value())

        # 회전 변화량 큰지 미리 경고 (이웃 점 사이)
        max_rot = 0.0
        for i in range(1, len(self.path_points)):
            rc = self._rot_change_deg(self.path_points[i - 1], self.path_points[i])
            if rc is not None and rc > max_rot:
                max_rot = rc

        msg = (
            f"▶ 경로 추적 실행\n\n"
            f"경로점: {len(self.path_points)}개 (LIN 직선)\n"
            f"속도: {speed}%" + (" (AUT 50% 상한)" if self._is_aut_mode() else "") + "\n"
            f"Sampling: {self.sampling_spin.value():.1f} mm\n"
            f"Offset:  {self.offset_spin.value():+.1f} mm (법선 바깥)\n"
            f"이웃 회전 변화 최대: {max_rot:.0f}°\n\n"
            f"⚠ T1 모드 — SmartPAD 데드맨+시작 버튼 필요\n"
            f"⚠ 비상시 Space 또는 비상정지 버튼\n\n"
            f"진행하시겠습니까?"
        )
        ret = QMessageBox.question(self, "경로 추적 확인", msg,
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"속도 설정 실패: {e}")
            return

        # 동적 큐 채움 시작
        self._pending_points = list(self.path_points)
        self._sent_count = 0
        self._total_count = len(self.path_points)
        self._motion_running = True

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_compute_path.setEnabled(False)

        self._update_progress()

        # QTimer로 비차단 폴링 (300ms 간격)
        self._send_timer = QTimer(self)
        self._send_timer.setInterval(300)
        self._send_timer.timeout.connect(self._try_send_next)
        self._send_timer.start()

        self.main.statusBar().showMessage("▶ 경로 추적 시작 — KRL 큐 동적 채움")

    def _try_send_next(self):
        """QTimer 폴링: 빈 슬롯 있으면 다음 경로점 1개 큐에 추가."""
        if not self._motion_running:
            return
        robot = self.main.robot
        if robot is None:
            self._finish_motion(error="로봇 연결 끊김")
            return

        if not self._pending_points:
            # 모든 점 송신 완료 → 큐가 빌 때까지 별도 폴링 후 종료
            self._send_timer.stop()
            self._wait_for_completion()
            return

        try:
            has_empty = robot.has_empty_slot()
        except Exception as e:
            logger.warning(f"슬롯 확인 실패: {e}")
            return
        if not has_empty:
            return  # 다음 cycle

        pt = self._pending_points[0]
        try:
            slot = robot.add_move_lin(pt["x"], pt["y"], pt["z"], pt["a"], pt["b"], pt["c"])
        except Exception as e:
            logger.error(f"점 전송 실패: {e}")
            self._finish_motion(error=str(e))
            return

        if slot is None:
            # 다음 cycle 재시도
            return

        self._pending_points.pop(0)
        self._sent_count += 1
        self._update_progress()
        logger.info(f"경로점 {self._sent_count}/{self._total_count} → slot={slot}")

    def _wait_for_completion(self):
        """모든 점을 큐에 보낸 후, KRL 큐가 빌 때까지 폴링."""
        if self._completion_timer is None:
            self._completion_timer = QTimer(self)
            self._completion_timer.setInterval(500)
            self._completion_timer.timeout.connect(self._check_completion)
        self._completion_timer.start()
        self.main.statusBar().showMessage(
            f"✅ 모든 점({self._total_count}) 송신 완료 — 큐가 비길 대기"
        )

    def _check_completion(self):
        robot = self.main.robot
        if robot is None:
            self._finish_motion(error="로봇 연결 끊김")
            return
        try:
            empty = robot.is_queue_empty()
        except Exception:
            return
        if empty:
            self._finish_motion()

    def _finish_motion(self, error: Optional[str] = None):
        self._motion_running = False
        if self._send_timer is not None:
            self._send_timer.stop()
        if self._completion_timer is not None:
            self._completion_timer.stop()

        self.btn_start.setEnabled(len(self.path_points) >= 2 and self.main.robot is not None)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_compute_path.setEnabled(self.start_pt is not None and self.end_pt is not None)

        if error:
            self.progress_label.setText(f"중단: {error}")
            self.main.statusBar().showMessage(f"⛔ 경로 추적 중단: {error}")
        else:
            self.progress_label.setText(
                f"완료: {self._sent_count}/{self._total_count}"
            )
            self.main.statusBar().showMessage("✅ 경로 추적 완료")

    def _stop_path_motion(self):
        """남은 점 전송 중단 + 큐 비움."""
        if not self._motion_running:
            return
        ret = QMessageBox.question(
            self, "정지 확인",
            "경로 추적을 중단하고 남은 KRL 큐를 비우시겠습니까?\n"
            "(이미 진행 중인 모션 1개는 끝까지 실행됩니다.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._pending_points.clear()
        try:
            self.main.robot.clear_queue()
        except Exception as e:
            logger.error(f"큐 비우기 실패: {e}")
        self._finish_motion(error="사용자 중단")

    def _update_progress(self):
        self.progress_label.setText(
            f"진행: {self._sent_count}/{self._total_count} 점 큐 전송"
            f" (남은 {len(self._pending_points)})"
        )

    def _rot_change_deg(self, a: Dict, b: Dict) -> Optional[float]:
        try:
            Ra = tcp_to_homogeneous(a)[:3, :3]
            Rb = tcp_to_homogeneous(b)[:3, :3]
            R = Ra.T @ Rb
            cos_a = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_a)))
        except Exception:
            return None

    # ------------------------------------------------------------
    # 모드 표시 (BinPickingTab과 동일)
    # ------------------------------------------------------------

    def _refresh_mode_display(self):
        if self.main.robot is None:
            self._current_mode = "?"
            self.mode_label.setText("모드: 미연결")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; "
                                          "background-color: #BDBDBD; color: white; border-radius: 3px;")
            return
        try:
            m = self.main.robot.read_variable("$MODE_OP")
            if m:
                self._current_mode = normalize_robot_mode(m)
        except Exception:
            return

        if is_auto_mode(self._current_mode):
            self.mode_label.setText(f"⚠ {self._current_mode} (자동 운용)")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; "
                                          "background-color: #D32F2F; color: white; border-radius: 3px;")
        elif "T1" in self._current_mode or "T2" in self._current_mode:
            self.mode_label.setText(f"{self._current_mode} (수동)")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; "
                                          "background-color: #2E7D32; color: white; border-radius: 3px;")
        else:
            self.mode_label.setText(f"모드: {self._current_mode}")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; "
                                          "background-color: #757575; color: white; border-radius: 3px;")
