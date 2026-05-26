"""
Bin Picking 탭
- 2D 이미지 뷰: 캡처, 마우스 드래그 ROI, vidnn 객체 탐지 bbox 표시
- 3D 포인트 클라우드 뷰 (PyVista): ROI 박스, 검출 객체 마커, 마우스 클릭 선택
- Hand-eye calibration 적용하여 로봇 base 좌표계로 피킹 위치 변환
"""

import sys
import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict

from PySide6.QtCore import Qt, QRect, QPoint, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QShortcut, QKeySequence
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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QApplication,
    QComboBox,
    QSplitter,
    QCheckBox,
    QListWidget,
    QScrollArea,
)

import pyvista as pv
from pyvistaqt import QtInteractor
import open3d as o3d

# vidnn 경로 추가
# 환경변수 VIDNN_PATH 로 override 가능 (다른 환경에서 클론 시 ~/.bashrc 등에 설정).
VIDNN_PATH = os.environ.get("VIDNN_PATH", "/home/robotegra/michael/vidnn")
if VIDNN_PATH not in sys.path:
    sys.path.insert(0, VIDNN_PATH)

from calibration import tcp_to_homogeneous
from kuka_robot import normalize_robot_mode, is_auto_mode
from robot_control_mixin import RobotControlMixin

logger = logging.getLogger(__name__)


# ============================================================
# 2D 이미지 뷰 (드래그 ROI + bbox 오버레이)
# ============================================================


class DraggableImageLabel(QLabel):
    """마우스 드래그로 ROI 선택 + 클릭으로 객체 선택 가능한 이미지 라벨"""

    roiChanged = Signal(int, int, int, int)  # x1, y1, x2, y2 (이미지 원본 좌표)
    objectPicked = Signal(int)  # 클릭된 객체 인덱스

    CLICK_THRESHOLD = 8  # 이 거리 이하로 이동하면 클릭으로 처리

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #2a2a2a; color: #888;")
        self.setMouseTracking(True)

        self._original_bgr = None
        # [(x1, y1, x2, y2, color, label, obj_index), ...]
        self._overlay_boxes = []
        self._roi_rect = None  # (x1, y1, x2, y2) 원본 이미지 좌표계
        self._highlighted_idx = None

        # 드래그 상태
        self._dragging = False
        self._drag_start = None
        self._drag_current = None

        # 표시 중인 스케일/오프셋 (원본 이미지 → 라벨 표시)
        self._display_scale = 1.0
        self._display_offset = (0, 0)
        self._display_size = (0, 0)  # 스케일된 이미지 크기

    def set_image(self, bgr: Optional[np.ndarray]):
        """이미지 설정 (BGR numpy array)"""
        self._original_bgr = bgr
        self._refresh()

    def set_boxes(self, boxes: List[tuple]):
        """검출된 bbox 표시: [(x1, y1, x2, y2, (r,g,b), label, obj_index), ...]"""
        self._overlay_boxes = boxes
        self._refresh()

    def set_highlight(self, idx: Optional[int]):
        """특정 객체 번호를 강조 (다른 것은 어둡게)"""
        self._highlighted_idx = idx
        self._refresh()

    def set_roi(self, rect: Optional[tuple]):
        """외부에서 ROI 설정 (x1, y1, x2, y2) 또는 None으로 해제"""
        self._roi_rect = rect
        self._refresh()

    def clear_all(self):
        """이미지/오버레이 전부 초기화 (QLabel.clear와 구분)"""
        self._original_bgr = None
        self._overlay_boxes = []
        self._roi_rect = None
        self.setText("이미지 없음")
        self.setPixmap(QPixmap())

    def _refresh(self):
        if self._original_bgr is None:
            self.setText("이미지 없음")
            return

        rgb = cv2.cvtColor(self._original_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        label_w = self.width()
        label_h = self.height()
        if label_w <= 0 or label_h <= 0:
            return

        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        self._display_scale = scale
        self._display_size = (new_w, new_h)
        self._display_offset = ((label_w - new_w) // 2, (label_h - new_h) // 2)

        # overlay 그리기 (원본 좌표계 → 스케일 적용하여 그리기)
        overlay = resized.copy()
        for box in self._overlay_boxes:
            # box: (x1, y1, x2, y2, color, label, obj_index)
            x1, y1, x2, y2, color, label = box[:6]
            obj_idx = box[6] if len(box) > 6 else None
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)

            # 강조 여부에 따른 스타일
            if self._highlighted_idx is not None and obj_idx == self._highlighted_idx:
                box_color = (0, 255, 0)  # 선택: 녹색
                thickness = 3
            elif self._highlighted_idx is not None:
                # 선택된 게 따로 있는 경우, 나머지는 어둡게
                dim = tuple(int(c * 0.5) for c in color)
                box_color = dim
                thickness = 2
            else:
                box_color = color
                thickness = 2

            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), box_color, thickness)

            # 번호 + 클래스명 라벨
            num_str = f"#{obj_idx + 1}" if obj_idx is not None else ""
            full_label = f"{num_str} {label}".strip() if label else num_str

            if full_label:
                (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(sy1 - 4, th + 2)
                cv2.rectangle(overlay, (sx1, ty - th - 4), (sx1 + tw + 4, ty + 2), box_color, -1)
                cv2.putText(overlay, full_label, (sx1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ROI (노란색 점선 스타일)
        if self._roi_rect is not None:
            rx1, ry1, rx2, ry2 = self._roi_rect
            sx1, sy1 = int(rx1 * scale), int(ry1 * scale)
            sx2, sy2 = int(rx2 * scale), int(ry2 * scale)
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 255, 0), 2)
            cv2.putText(overlay, "ROI", (sx1, max(sy1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        qimage = QImage(overlay.data, new_w, new_h, new_w * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)

        # 드래그 중인 사각형 오버레이 (위젯 좌표계에 직접 그리기)
        if self._dragging and self._drag_start and self._drag_current:
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            ox, oy = self._display_offset
            p1 = QPoint(self._drag_start.x() - ox, self._drag_start.y() - oy)
            p2 = QPoint(self._drag_current.x() - ox, self._drag_current.y() - oy)
            painter.drawRect(QRect(p1, p2).normalized())
            painter.end()

        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _widget_to_image(self, pt: QPoint) -> Optional[tuple]:
        """위젯 좌표 → 원본 이미지 좌표 변환"""
        if self._original_bgr is None or self._display_scale <= 0:
            return None
        ox, oy = self._display_offset
        nw, nh = self._display_size
        x = pt.x() - ox
        y = pt.y() - oy
        if not (0 <= x < nw and 0 <= y < nh):
            return None
        return int(x / self._display_scale), int(y / self._display_scale)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._original_bgr is not None:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_current = event.position().toPoint()
            self._refresh()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            end = event.position().toPoint()
            start_pt = self._drag_start
            self._drag_start = None
            self._drag_current = None

            if start_pt is None:
                self._refresh()
                return

            # 드래그 거리로 클릭/드래그 구분
            dx = end.x() - start_pt.x()
            dy = end.y() - start_pt.y()
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= self.CLICK_THRESHOLD:
                # 클릭으로 처리: 해당 픽셀이 어느 bbox에 들어있는지 찾기
                click_img = self._widget_to_image(end)
                if click_img and self._overlay_boxes:
                    cx, cy = click_img
                    hit_idx = None
                    hit_area = float("inf")
                    for box in self._overlay_boxes:
                        x1, y1, x2, y2 = box[:4]
                        obj_idx = box[6] if len(box) > 6 else None
                        if obj_idx is None:
                            continue
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            area = (x2 - x1) * (y2 - y1)
                            # 겹치는 경우 작은 bbox 우선
                            if area < hit_area:
                                hit_area = area
                                hit_idx = obj_idx
                    if hit_idx is not None:
                        self.objectPicked.emit(hit_idx)
                self._refresh()
                return

            # 드래그: ROI 선택
            start_img = self._widget_to_image(start_pt)
            end_img = self._widget_to_image(end)
            if start_img and end_img:
                x1, y1 = start_img
                x2, y2 = end_img
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self._roi_rect = (x1, y1, x2, y2)
                    self.roiChanged.emit(x1, y1, x2, y2)
            self._refresh()


# ============================================================
# 3D 포인트 클라우드 뷰 (PyVista)
# ============================================================


class PointCloudView3D(QWidget):
    """PyVista 기반 3D 포인트 클라우드 뷰 + 객체 클릭 선택"""

    objectPicked = Signal(int)  # 선택된 객체 인덱스

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        layout.addWidget(self.plotter)

        self._object_actors = {}  # name → actor
        self._object_centers = {}  # name → (idx, center)
        self._all_marker_names = []  # 검출마다 지울 대상 (sphere/arrow/label)
        self._picking_initialized = False
        self._saved_camera = None  # (focal, vfov_deg) 저장

        # 마우스 픽킹은 한 번만 활성화
        self._enable_picking_once()

    def _enable_picking_once(self):
        if self._picking_initialized:
            return
        try:
            self.plotter.enable_point_picking(
                callback=self._on_point_pick,
                use_picker=True,  # VTK picker 인스턴스 전달받음
                show_message=False,
                show_point=False,
                left_clicking=True,
            )
            self._picking_initialized = True
        except Exception as e:
            logger.warning(f"point picking 활성화 실패: {e}")

    def clear(self):
        self.plotter.clear()
        self._object_actors.clear()
        self._object_centers.clear()
        self._all_marker_names.clear()

    def show_pointcloud(
        self, xyz: np.ndarray, rgb: Optional[np.ndarray] = None, intrinsics: Optional[np.ndarray] = None, image_shape: Optional[tuple] = None
    ):
        """
        포인트 클라우드 표시 + 카메라 시점 자동 설정

        Args:
            xyz: (H, W, 3) 또는 (N, 3) 카메라 좌표계 (mm)
            rgb: (H, W, 3) 또는 (N, 3) 0~255 uint8
            intrinsics: 3x3 내부 파라미터 (있으면 정확한 화각/주점 반영)
            image_shape: (H, W) 이미지 크기 (있으면 정확한 화각 계산)
        """
        if xyz.ndim == 3:
            pts = xyz.reshape(-1, 3)
            if image_shape is None:
                image_shape = xyz.shape[:2]
        else:
            pts = xyz

        mask = ~np.any(np.isnan(pts), axis=1)
        pts = pts[mask]

        if len(pts) == 0:
            return

        cloud = pv.PolyData(pts.astype(np.float32))

        if rgb is not None:
            if rgb.ndim == 3:
                colors = rgb.reshape(-1, 3)
            else:
                colors = rgb
            colors = colors[mask]
            cloud["colors"] = colors.astype(np.uint8)
            self.plotter.add_mesh(
                cloud, scalars="colors", rgb=True, point_size=2, render_points_as_spheres=False, name="pointcloud", pickable=False, reset_camera=False
            )
        else:
            self.plotter.add_mesh(
                cloud, color="lightgray", point_size=2, render_points_as_spheres=False, name="pointcloud", pickable=False, reset_camera=False
            )

        # 좌표축 (100mm 크기)
        self.plotter.add_axes_at_origin(labels_off=False, line_width=3, x_color="red", y_color="green", z_color="blue")

        # 카메라 시점 설정 (2D 이미지와 동일한 화각 복제) - VTK 직접 조작
        self.set_camera_from_intrinsics(pts, intrinsics, image_shape)

    def set_camera_from_intrinsics(self, pts: np.ndarray, intrinsics: Optional[np.ndarray], image_shape: Optional[tuple]):
        """Zivid intrinsics를 이용해 2D 이미지와 동일한 화각으로 카메라 설정 (VTK 직접 조작)"""
        # 다음 render 시 적용할 카메라 파라미터 저장 (탭 전환 시 재적용용)
        focus_z = float(np.median(pts[:, 2]))
        if focus_z <= 0:
            focus_z = 1000.0

        vfov_deg = 45.0
        focal = (0.0, 0.0, focus_z)

        if intrinsics is not None and image_shape is not None:
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
            h, w = image_shape[:2]

            vfov_deg = float(np.degrees(2 * np.arctan(h / (2 * fy))))
            nx = (w / 2 - cx) / fx
            ny = (h / 2 - cy) / fy
            focal = (nx * focus_z, ny * focus_z, focus_z)

        self._saved_camera = (focal, vfov_deg)
        self._apply_camera()

    def _apply_camera(self):
        """저장된 카메라 파라미터를 VTK에 직접 적용"""
        if not hasattr(self, "_saved_camera") or self._saved_camera is None:
            return
        focal, vfov_deg = self._saved_camera

        cam = self.plotter.camera
        cam.SetParallelProjection(False)
        cam.SetPosition(0.0, 0.0, 0.0)
        cam.SetFocalPoint(float(focal[0]), float(focal[1]), float(focal[2]))
        cam.SetViewUp(0.0, -1.0, 0.0)
        cam.SetViewAngle(float(vfov_deg))  # degrees

        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def refresh_camera(self):
        """외부에서 호출 가능 - 위젯 크기 변경 후 재적용"""
        self._apply_camera()

    def show_roi_box(self, bounds: tuple):
        """ROI 박스 표시 (x_min, x_max, y_min, y_max, z_min, z_max) - 외곽선만"""
        box = pv.Box(bounds=bounds)
        edges = box.extract_feature_edges(feature_angle=30)
        self.plotter.add_mesh(edges, color="yellow", line_width=4, name="roi_box", pickable=False, render_lines_as_tubes=True, reset_camera=False)

    def show_pick_objects(self, objects: List[Dict]):
        """
        피킹 객체들 표시 (배치 렌더링)

        - 모든 sphere를 하나의 glyph로 한 번에 추가
        - 모든 arrow를 하나의 glyph로 한 번에 추가
        - 라벨도 한 번의 호출로 모두 추가
        - Picking은 point_picking으로 클릭 좌표에서 가장 가까운 객체 탐색
        """
        # 기존 마커 제거
        for name in self._all_marker_names:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self._all_marker_names.clear()
        self._object_actors.clear()
        self._object_centers.clear()

        if not objects:
            self.plotter.render()
            return

        centers = np.array([obj["center"] for obj in objects], dtype=np.float32)
        normals = np.array([obj["normal"] for obj in objects], dtype=np.float32)
        indices = [obj["index"] for obj in objects]
        class_names = [obj["class_name"] for obj in objects]

        # Picking용 중심 저장
        for idx, center in zip(indices, centers):
            self._object_centers[idx] = np.array(center)

        # 1) Sphere glyph (하나의 mesh로 N개 구 배치)
        center_cloud = pv.PolyData(centers)
        sphere_glyph = center_cloud.glyph(
            geom=pv.Sphere(radius=6),
            scale=False,
            orient=False,
        )
        actor_s = self.plotter.add_mesh(
            sphere_glyph,
            color="red",
            name="pick_spheres",
            pickable=True,
            reset_camera=False,
        )
        self._object_actors["pick_spheres"] = actor_s
        self._all_marker_names.append("pick_spheres")

        # 2) Arrow glyph (법선 방향, 모두 한 번에)
        #    각 방향 벡터를 polydata에 할당하여 glyph의 orient 사용
        arrow_cloud = pv.PolyData(centers)
        arrow_cloud["vectors"] = (-normals * 40.0).astype(np.float32)
        arrow_glyph = arrow_cloud.glyph(
            geom=pv.Arrow(),
            orient="vectors",
            scale="vectors",
            factor=1.0,
        )
        self.plotter.add_mesh(
            arrow_glyph,
            color="cyan",
            name="pick_arrows",
            pickable=False,
            reset_camera=False,
        )
        self._all_marker_names.append("pick_arrows")

        # 3) 라벨 (한 번의 호출로 전체)
        label_points = centers + np.array([0, 0, -10], dtype=np.float32)
        label_texts = [f"#{idx + 1} {name}" for idx, name in zip(indices, class_names)]
        self.plotter.add_point_labels(
            label_points,
            label_texts,
            point_size=1,
            font_size=14,
            text_color="white",
            name="pick_labels",
            always_visible=True,
            pickable=False,
            show_points=False,
        )
        self._all_marker_names.append("pick_labels")

        self.plotter.render()

        # 픽킹은 __init__에서 한 번만 활성화됨

    def highlight(self, idx: int):
        """선택된 객체 강조 (녹색 구를 위에 덮어씌움)"""
        try:
            self.plotter.remove_actor("highlight_sphere")
        except Exception:
            pass

        if idx in self._object_centers:
            center = self._object_centers[idx]
            hl = pv.Sphere(radius=7, center=center)
            self.plotter.add_mesh(
                hl,
                color="green",
                name="highlight_sphere",
                pickable=False,
                reset_camera=False,
            )
        self.plotter.render()

    def reset_view(self):
        self.plotter.reset_camera()

    def _on_point_pick(self, *args):
        """
        point picking 콜백
        use_picker=True 이므로 (picked_xyz, picker)가 전달됨
        """
        if not args:
            return
        picked = args[0]
        if picked is None:
            return
        picked_arr = np.asarray(picked, dtype=np.float32)
        if picked_arr.shape != (3,):
            return

        # 가장 가까운 객체 중심 찾기
        best_idx = None
        best_dist = float("inf")
        for obj_idx, center in self._object_centers.items():
            d = float(np.linalg.norm(picked_arr - center))
            if d < best_dist:
                best_dist = d
                best_idx = obj_idx

        # sphere 반경(6) + 여유를 고려해 15mm 이내만 인정
        if best_idx is not None and best_dist < 15:
            self.highlight(best_idx)
            self.objectPicked.emit(best_idx)


# ============================================================
# Bin Picking 탭
# ============================================================


class BinPickingTab(RobotControlMixin, QWidget):
    """
    Bin Picking 통합 탭
    - 캡처 → 객체 탐지 → 포즈 계산 → 로봇 좌표 변환
    """

    # 환경변수 VIDNN_MODEL_PATH 로 override 가능 (다른 환경에서 다른 모델 사용).
    VIDNN_MODEL_PATH = os.environ.get(
        "VIDNN_MODEL_PATH", "/home/robotegra/michael/vidnn/runs/ladybug.pt"
    )

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        # 상태
        self.current_image = None  # BGR
        self.current_xyz = None  # (H, W, 3) mm
        self.current_rgb = None  # (H, W, 3) uint8 (PyVista 용)
        self.current_normals = None  # (H, W, 3) Zivid normals
        self.current_intrinsics = None  # 3x3
        self.roi_2d = None  # (x1, y1, x2, y2) 픽셀
        self.roi_3d = None  # {x_min, x_max, y_min, y_max, z_min, z_max} mm
        self.detections = []  # vidnn 탐지 리스트
        self.pick_objects = []  # 포즈 계산된 객체
        self.selected_idx = None
        self.target_pose = None  # 선택된 객체의 로봇 base 좌표계 자세

        self.T_calib = None
        self.calib_mode = None

        # 시퀀스 큐 (Python에서 만드는 액션 시나리오)
        # 각 액션: {"type": "object_move"|"home", "label": str, "target": dict, ...}
        self.user_queue = []

        # 선택된 객체의 TCP 자세 시각화 actor 이름 추적 (이전 시각화 정확히 제거용)
        self._tcp_viz_actors = []

        # 현재 로봇 모드 캐시
        self._current_mode = "?"

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === 상단: 컨트롤 행 ===
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
        top_row.addWidget(self.btn_clear_roi)

        self.btn_detect = QPushButton("객체 검출")
        self.btn_detect.clicked.connect(self._detect)
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        top_row.addWidget(self.btn_detect)

        top_row.addSpacing(15)
        top_row.addWidget(QLabel("Conf:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setFixedWidth(70)
        top_row.addWidget(self.conf_spin)

        top_row.addStretch()

        # 로봇 모드 표시 라벨 (2초마다 자동 갱신)
        self.mode_label = QLabel("모드: ?")
        self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; " "background-color: #BDBDBD; color: white; border-radius: 3px;")
        top_row.addWidget(self.mode_label)

        # 뷰 스위치 버튼
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

        # === 중앙: 2D/3D 스택 + 정보 패널 ===
        splitter = QSplitter(Qt.Horizontal)

        self.view_stack = QStackedWidget()
        self.view_2d = DraggableImageLabel()
        self.view_2d.roiChanged.connect(self._on_roi_dragged)
        self.view_2d.objectPicked.connect(self._on_object_picked)
        self.view_stack.addWidget(self.view_2d)

        self.view_3d = PointCloudView3D()
        self.view_3d.objectPicked.connect(self._on_object_picked)
        self.view_stack.addWidget(self.view_3d)

        splitter.addWidget(self.view_stack)

        # 우측 정보 패널
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # 검출 리스트
        det_group = QGroupBox("검출된 객체")
        det_layout = QVBoxLayout(det_group)
        self.det_table = QTableWidget(0, 3)
        self.det_table.setHorizontalHeaderLabels(["#", "클래스", "Conf"])
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.det_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.det_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.det_table.itemSelectionChanged.connect(self._on_table_selection)
        det_layout.addWidget(self.det_table)
        info_layout.addWidget(det_group)

        # 선택된 객체 정보
        sel_group = QGroupBox("선택된 피킹 포즈 (로봇 base 좌표)")
        sel_layout = QVBoxLayout(sel_group)
        self.robot_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 15px; font-weight: bold; color: #0066cc;")
            self.robot_labels[axis] = lab
            row.addWidget(lab)
            row.addStretch()
            sel_layout.addLayout(row)
        info_layout.addWidget(sel_group)

        # 로봇 이동 제어
        move_group = QGroupBox("로봇 이동 제어")
        move_layout = QVBoxLayout(move_group)

        # 이동 방식 + 속도
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("방식:"))
        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(["LIN (직선, 추천)", "PTP (최단 경로)"])
        opt_row.addWidget(self.move_mode_combo)
        move_layout.addLayout(opt_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("속도(%):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(30)
        self.speed_spin.setFixedWidth(70)
        # 값 변경 시 즉시 로봇에 적용
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_spin)
        self.btn_apply_speed = QPushButton("적용")
        self.btn_apply_speed.setFixedWidth(50)
        self.btn_apply_speed.clicked.connect(self._apply_speed_now)
        speed_row.addWidget(self.btn_apply_speed)
        speed_row.addStretch()
        move_layout.addLayout(speed_row)

        # 접근/철수 옵션
        approach_row = QHBoxLayout()
        self.use_approach = QCheckBox("접근/철수 사용")
        self.use_approach.setChecked(True)
        self.use_approach.setToolTip(
            "체크 시 [Approach → Target → Retract] 3단계 모션을 큐에 추가\n" "법선 방향으로 위에 안전하게 다가갔다 → 정밀 접근 → 다시 위로"
        )
        approach_row.addWidget(self.use_approach)
        approach_row.addWidget(QLabel("거리(mm):"))
        self.approach_dist = QSpinBox()
        self.approach_dist.setRange(5, 500)
        self.approach_dist.setValue(50)
        self.approach_dist.setFixedWidth(60)
        approach_row.addWidget(self.approach_dist)
        approach_row.addStretch()
        move_layout.addLayout(approach_row)

        # Z 최소 한계 (안전: 바닥 충돌 방지)
        zlim_row = QHBoxLayout()
        zlim_row.addWidget(QLabel("Z 최소(mm):"))
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(-2000, 2000)
        self.z_min_spin.setValue(5)
        self.z_min_spin.setFixedWidth(80)
        self.z_min_spin.setToolTip("타겟 Z 좌표가 이 값보다 낮으면 이동을 거부합니다 (바닥 충돌 방지)")
        zlim_row.addWidget(self.z_min_spin)
        zlim_row.addStretch()
        move_layout.addLayout(zlim_row)

        # 이동 버튼
        self.btn_move_robot = QPushButton("선택 위치로 이동")
        self.btn_move_robot.setMinimumHeight(45)
        self.btn_move_robot.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1976D2; color: white;")
        self.btn_move_robot.clicked.connect(self._execute_move)
        self.btn_move_robot.setEnabled(False)
        move_layout.addWidget(self.btn_move_robot)

        # Home 이동/재설정 버튼 (한 줄에 배치)
        home_row = QHBoxLayout()

        self.btn_move_home = QPushButton("🏠 Home으로 이동")
        self.btn_move_home.setMinimumHeight(40)
        self.btn_move_home.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #2E7D32; color: white;")
        self.btn_move_home.clicked.connect(self._move_to_home)
        self.btn_move_home.setEnabled(False)
        home_row.addWidget(self.btn_move_home, stretch=2)

        self.btn_set_home = QPushButton("📍 Home\n재설정")
        self.btn_set_home.setMinimumHeight(40)
        self.btn_set_home.setStyleSheet("font-size: 11px; background-color: #689F38; color: white;")
        self.btn_set_home.setToolTip("현재 로봇 TCP 위치를 새 Home으로 저장합니다")
        self.btn_set_home.clicked.connect(self._set_home_to_current)
        self.btn_set_home.setEnabled(False)
        home_row.addWidget(self.btn_set_home, stretch=1)

        move_layout.addLayout(home_row)

        # 큐 비우기 버튼 (이전 명령 취소)
        self.btn_clear_queue = QPushButton("🗑 큐 비우기 (이전 명령 취소)")
        self.btn_clear_queue.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold;")
        self.btn_clear_queue.clicked.connect(self._clear_motion_queue)
        move_layout.addWidget(self.btn_clear_queue)

        # 비상정지 버튼 (큼지막한 빨간색)
        self.btn_estop = QPushButton("⛔ 비상정지 (Space)")
        self.btn_estop.setMinimumHeight(60)
        self.btn_estop.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #D32F2F; color: white;")
        self.btn_estop.clicked.connect(self._emergency_stop)
        move_layout.addWidget(self.btn_estop)

        # 비상정지 해제 버튼 (작게)
        self.btn_estop_release = QPushButton("비상정지 해제")
        self.btn_estop_release.setStyleSheet("background-color: #757575; color: white;")
        self.btn_estop_release.clicked.connect(self._emergency_stop_release)
        move_layout.addWidget(self.btn_estop_release)

        info_layout.addWidget(move_group)

        # === 시퀀스 큐 (자동 실행 시나리오) ===
        seq_group = QGroupBox("시퀀스 큐 (자동 실행 순서)")
        seq_layout = QVBoxLayout(seq_group)

        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(80)
        self.action_list.setMaximumHeight(150)
        seq_layout.addWidget(self.action_list)

        # 추가 버튼들
        add_row = QHBoxLayout()
        self.btn_add_obj_to_seq = QPushButton("➕ 객체 이동 추가")
        self.btn_add_obj_to_seq.setStyleSheet("background-color: #1976D2; color: white;")
        self.btn_add_obj_to_seq.clicked.connect(self._enqueue_object_move)
        self.btn_add_obj_to_seq.setEnabled(False)
        add_row.addWidget(self.btn_add_obj_to_seq)

        self.btn_add_home_to_seq = QPushButton("➕ Home 추가")
        self.btn_add_home_to_seq.setStyleSheet("background-color: #2E7D32; color: white;")
        self.btn_add_home_to_seq.clicked.connect(self._enqueue_home_to_sequence)
        self.btn_add_home_to_seq.setEnabled(False)
        add_row.addWidget(self.btn_add_home_to_seq)
        seq_layout.addLayout(add_row)

        # 제거 버튼들
        del_row = QHBoxLayout()
        self.btn_remove_seq_item = QPushButton("선택 항목 제거")
        self.btn_remove_seq_item.clicked.connect(self._remove_selected_action)
        del_row.addWidget(self.btn_remove_seq_item)

        self.btn_clear_seq = QPushButton("시퀀스 비우기")
        self.btn_clear_seq.clicked.connect(self._clear_user_queue)
        del_row.addWidget(self.btn_clear_seq)
        seq_layout.addLayout(del_row)

        # 시작 버튼 (큰 파란색)
        self.btn_start_seq = QPushButton("▶ 시퀀스 시작")
        self.btn_start_seq.setMinimumHeight(45)
        self.btn_start_seq.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1565C0; color: white;")
        self.btn_start_seq.clicked.connect(self._start_sequence)
        seq_layout.addWidget(self.btn_start_seq)

        info_layout.addWidget(seq_group)

        info_layout.addStretch()

        # 우측 패널을 ScrollArea로 감싸서 모든 컨트롤이 잘리지 않게 함
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

        # 스페이스바 = 비상정지 (탭이 활성일 때만 동작)
        sc_estop = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_estop.setContext(Qt.WidgetWithChildrenShortcut)
        sc_estop.activated.connect(self._emergency_stop)

        # 모드 표시 자동 갱신 (2초마다)
        self._mode_timer = QTimer(self)
        self._mode_timer.timeout.connect(self._refresh_mode_display)
        self._mode_timer.start(2000)

    def _switch_view(self, idx: int):
        self.view_stack.setCurrentIndex(idx)
        self.btn_view_2d.setChecked(idx == 0)
        self.btn_view_3d.setChecked(idx == 1)
        # 3D로 전환 시 위젯 크기가 확정된 후 카메라 재적용
        if idx == 1:
            # Qt가 리사이즈를 처리한 뒤 카메라 적용 (지연)
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self.view_3d.refresh_camera)

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

        image = self.main.camera.frame_to_2d_image(frame)  # BGR
        xyz = self.main.camera.frame_to_point_cloud(frame)  # (H, W, 3) mm
        normals = self.main.camera.frame_to_normals(frame)  # (H, W, 3)
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

        # 검출 리셋
        self.detections = []
        self.pick_objects = []
        self.selected_idx = None
        self.target_pose = None
        self._tcp_viz_actors.clear()  # 3D 뷰가 곧 clear되니 추적 리스트도 비움
        self.btn_move_robot.setEnabled(False)
        self.det_table.setRowCount(0)
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.robot_labels[axis].setText("---")

        # 2D 뷰 갱신
        self.view_2d.set_image(image)
        self.view_2d.set_boxes([])
        self.view_2d.set_highlight(None)

        # 3D 뷰 갱신 (2D 이미지와 동일한 화각)
        self.view_3d.clear()
        self.view_3d.show_pointcloud(
            xyz,
            self.current_rgb,
            intrinsics=self.current_intrinsics,
            image_shape=image.shape,
        )
        if self.roi_3d is not None:
            self._apply_roi_to_3d()
        self.view_3d.reset_view()

        self.main.statusBar().showMessage("캡처 완료")

    def _on_roi_dragged(self, x1: int, y1: int, x2: int, y2: int):
        self.roi_2d = (x1, y1, x2, y2)

        # 2D bbox 안의 3D 포인트들의 범위 → 3D ROI
        if self.current_xyz is not None:
            h, w = self.current_xyz.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            region = self.current_xyz[y1:y2, x1:x2].reshape(-1, 3)
            valid = region[~np.any(np.isnan(region), axis=1)]
            if len(valid) > 10:
                self.roi_3d = {
                    "x_min": float(valid[:, 0].min()),
                    "x_max": float(valid[:, 0].max()),
                    "y_min": float(valid[:, 1].min()),
                    "y_max": float(valid[:, 1].max()),
                    "z_min": float(valid[:, 2].min()),
                    "z_max": float(valid[:, 2].max()),
                }
                self._apply_roi_to_3d()
                self.main.statusBar().showMessage(
                    f"ROI 설정: X[{self.roi_3d['x_min']:.0f},{self.roi_3d['x_max']:.0f}] "
                    f"Y[{self.roi_3d['y_min']:.0f},{self.roi_3d['y_max']:.0f}] "
                    f"Z[{self.roi_3d['z_min']:.0f},{self.roi_3d['z_max']:.0f}] mm"
                )

    def _apply_roi_to_3d(self):
        if self.roi_3d is None:
            return
        bounds = (
            self.roi_3d["x_min"],
            self.roi_3d["x_max"],
            self.roi_3d["y_min"],
            self.roi_3d["y_max"],
            self.roi_3d["z_min"],
            self.roi_3d["z_max"],
        )
        self.view_3d.show_roi_box(bounds)

    def _clear_roi(self):
        self.roi_2d = None
        self.roi_3d = None
        self.view_2d.set_roi(None)
        try:
            self.view_3d.plotter.remove_actor("roi_box")
        except Exception:
            pass
        self.main.statusBar().showMessage("ROI 해제")

    def _detect(self):
        if self.current_image is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return

        self.main.statusBar().showMessage("객체 탐지 중...")
        QApplication.processEvents()

        try:
            from vidnn.module.inference import Predictor
        except ImportError as e:
            QMessageBox.critical(self, "오류", f"vidnn import 실패:\n{e}")
            return

        if not os.path.exists(self.VIDNN_MODEL_PATH):
            QMessageBox.critical(self, "오류", f"모델 파일 없음:\n{self.VIDNN_MODEL_PATH}")
            return

        try:
            # Predictor 캐싱
            if not hasattr(self, "_predictor"):
                self._predictor = Predictor(
                    model_path=self.VIDNN_MODEL_PATH,
                    task="detect",
                    conf=self.conf_spin.value(),
                    iou=0.6,
                    imgsz=640,
                )
            model = self._predictor
            model.conf = self.conf_spin.value()

            pred, _ = model(self.current_image)
            pred = pred.cpu().numpy()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"vidnn 추론 실패:\n{e}")
            return

        # 검출 결과 + ROI 필터링
        detections = []
        for row in pred:
            x1, y1, x2, y2 = row[:4]
            conf = float(row[4])
            cls_id = int(row[5])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

        # ROI 필터
        if self.roi_3d is not None:
            filtered = []
            for det in detections:
                bbox = det["bbox"]
                bcx = int((bbox[0] + bbox[2]) / 2)
                bcy = int((bbox[1] + bbox[3]) / 2)
                h, w = self.current_xyz.shape[:2]
                if 0 <= bcx < w and 0 <= bcy < h:
                    pt = self.current_xyz[bcy, bcx]
                    if not np.any(np.isnan(pt)):
                        in_roi = (
                            self.roi_3d["x_min"] <= pt[0] <= self.roi_3d["x_max"]
                            and self.roi_3d["y_min"] <= pt[1] <= self.roi_3d["y_max"]
                            and self.roi_3d["z_min"] <= pt[2] <= self.roi_3d["z_max"]
                        )
                        if in_roi:
                            filtered.append(det)
                            continue
            logger.info(f"ROI 필터링: {len(detections)} → {len(filtered)}")
            detections = filtered

        self.detections = detections

        # 각 검출에 대해 3D 포즈 계산 (먼저 실행해서 유효 객체만 표시)
        self._compute_pick_poses()

        # 2D 뷰에 bbox 표시 (유효한 피킹 포즈가 있는 객체만)
        colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80), (255, 80, 255), (80, 255, 255)]
        valid_indices = {o["index"] for o in self.pick_objects}
        boxes = []
        draw_i = 0  # 유효 객체만 순차 색 할당 (raw index 기준이면 팔레트가 듬성듬성 쓰여
        #            테이블/3D 순서와 색이 어긋남)
        for i, det in enumerate(detections):
            if i not in valid_indices:
                continue
            color = colors[draw_i % len(colors)]
            draw_i += 1
            label = f"{det['class_name']} {det['confidence']:.2f}"
            boxes.append((*det["bbox"], color, label, i))
        self.view_2d.set_boxes(boxes)

        # 테이블 갱신
        self._update_table()

        # 3D 뷰: 포인트클라우드는 유지하고 객체 마커만 갱신
        self.view_3d.show_pick_objects(self.pick_objects)

        self.main.statusBar().showMessage(f"검출 완료: {len(detections)}개, 피킹 포즈 {len(self.pick_objects)}개")

    def _compute_pick_poses(self):
        """
        각 검출 bbox의 3D 크롭 → 피킹 위치/법선 계산

        - 중심: bbox 안 유효 XYZ의 좌표 median (outlier에 강건)
        - 법선: 중심점을 2D로 투영한 픽셀 주변의 Zivid normals 평균
          (bbox 전체 평균은 배경 영향을 받으므로 중심 근방 패치만 사용)
        """
        self.pick_objects = []
        if self.current_xyz is None:
            return

        h, w = self.current_xyz.shape[:2]
        has_normals = self.current_normals is not None

        # intrinsics로 3D → 2D 투영용
        intr = self.current_intrinsics
        if intr is not None:
            fx, fy = float(intr[0, 0]), float(intr[1, 1])
            cx_i, cy_i = float(intr[0, 2]), float(intr[1, 2])
        else:
            fx = fy = cx_i = cy_i = None

        NORMAL_PATCH_RADIUS = 15  # 중심 주변 31x31 픽셀

        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = det["bbox"]
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(w, int(x2))
            iy2 = min(h, int(y2))

            crop_region = self.current_xyz[iy1:iy2, ix1:ix2].reshape(-1, 3)
            valid_mask = ~np.any(np.isnan(crop_region), axis=1)
            crop = crop_region[valid_mask]
            if len(crop) < 20:
                logger.warning(f"[{i}] 3D 포인트 부족: {len(crop)}")
                continue

            # 중심 = 좌표 median (outlier에 강건)
            center = np.median(crop, axis=0)

            # 중심점을 2D 픽셀로 투영 (법선 패치 위치용)
            if fx is not None and center[2] > 0:
                center_px = int(round(fx * center[0] / center[2] + cx_i))
                center_py = int(round(fy * center[1] / center[2] + cy_i))
            else:
                center_px = (ix1 + ix2) // 2
                center_py = (iy1 + iy2) // 2

            # bbox 범위로 클램프 (투영이 bbox 밖으로 나간 경우 안전망)
            center_px = max(ix1, min(ix2 - 1, center_px))
            center_py = max(iy1, min(iy2 - 1, center_py))

            # 법선 = 중심 주변 작은 패치의 Zivid normals 평균
            normal = None
            if has_normals:
                px1 = max(0, center_px - NORMAL_PATCH_RADIUS)
                px2 = min(w, center_px + NORMAL_PATCH_RADIUS + 1)
                py1 = max(0, center_py - NORMAL_PATCH_RADIUS)
                py2 = min(h, center_py + NORMAL_PATCH_RADIUS + 1)

                patch_n = self.current_normals[py1:py2, px1:px2].reshape(-1, 3)
                valid_n = patch_n[~np.any(np.isnan(patch_n), axis=1)]
                if len(valid_n) >= 3:
                    mean_n = valid_n.mean(axis=0)
                    nn = np.linalg.norm(mean_n)
                    if nn > 1e-6:
                        normal = mean_n / nn

            # Fallback: 중심 주변 XYZ 패치로 SVD 평면 피팅
            if normal is None:
                px1 = max(0, center_px - NORMAL_PATCH_RADIUS)
                px2 = min(w, center_px + NORMAL_PATCH_RADIUS + 1)
                py1 = max(0, center_py - NORMAL_PATCH_RADIUS)
                py2 = min(h, center_py + NORMAL_PATCH_RADIUS + 1)
                patch_xyz = self.current_xyz[py1:py2, px1:px2].reshape(-1, 3)
                patch_xyz = patch_xyz[~np.any(np.isnan(patch_xyz), axis=1)]
                if len(patch_xyz) >= 3:
                    normal = self._svd_normal(patch_xyz)
                else:
                    normal = np.array([0.0, 0.0, -1.0])

            # 법선은 카메라를 향하도록
            if normal[2] > 0:
                normal = -normal

            self.pick_objects.append(
                {
                    "index": i,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "center": center.tolist(),
                    "normal": normal.tolist(),
                    "n_points": int(len(crop)),
                }
            )

    @staticmethod
    def _svd_normal(pts: np.ndarray) -> np.ndarray:
        """Fallback: XYZ 포인트로 평면 피팅하여 법선 계산 (SVD)"""
        centered = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            return Vt[-1]
        except Exception:
            return np.array([0.0, 0.0, -1.0])

    def _update_table(self):
        self.det_table.setRowCount(0)
        for obj in self.pick_objects:
            row = self.det_table.rowCount()
            self.det_table.insertRow(row)
            self.det_table.setItem(row, 0, QTableWidgetItem(str(obj["index"] + 1)))
            self.det_table.setItem(row, 1, QTableWidgetItem(obj["class_name"]))
            self.det_table.setItem(row, 2, QTableWidgetItem(f"{obj['confidence']:.2f}"))

    def _on_table_selection(self):
        rows = self.det_table.selectionModel().selectedRows()
        if not rows:
            return
        row_idx = rows[0].row()
        if row_idx >= len(self.pick_objects):
            return
        obj = self.pick_objects[row_idx]
        self._select_object(obj["index"])

    def _on_object_picked(self, idx: int):
        self._select_object(idx)
        # 테이블에서도 해당 행 선택
        for row in range(self.det_table.rowCount()):
            item = self.det_table.item(row, 0)
            if item and int(item.text()) - 1 == idx:
                self.det_table.selectRow(row)
                break

    def _select_object(self, idx: int):
        """객체 선택 → 로봇 base 좌표 계산 및 표시"""
        self.selected_idx = idx
        self.view_3d.highlight(idx)
        self.view_2d.set_highlight(idx)

        obj = next((o for o in self.pick_objects if o["index"] == idx), None)
        if obj is None:
            return

        if self.T_calib is None:
            self.main.statusBar().showMessage("캘리브레이션 파일 로드 필요")
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                self.robot_labels[axis].setText("---")
            return

        center_cam = np.array(obj["center"])
        normal_cam = np.array(obj["normal"])

        # 카메라 좌표계 → 로봇 base 좌표계
        if self.calib_mode == "eye_to_hand":
            p_h = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0])
            center_base = (self.T_calib @ p_h)[:3]
            normal_base = self.T_calib[:3, :3] @ normal_cam
        elif self.calib_mode == "eye_in_hand":
            if not self.main.robot:
                QMessageBox.warning(self, "오류", "Eye-in-Hand 모드는 로봇 연결 필요")
                return
            cur_tcp = self.main.robot.get_tcp_position()
            T_g2b = tcp_to_homogeneous(cur_tcp)
            p_h = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0])
            center_base = (T_g2b @ self.T_calib @ p_h)[:3]
            normal_base = T_g2b[:3, :3] @ self.T_calib[:3, :3] @ normal_cam
        else:
            QMessageBox.critical(self, "오류", f"알 수 없는 모드: {self.calib_mode}")
            return

        # 법선 방향으로 TCP 자세 계산: Tool +Z가 법선 반대 방향 (표면을 향해)
        # 현재 TCP X축을 참고 자세로 사용 (없으면 월드 X축)
        if self.main.robot:
            cur_tcp = self.main.robot.get_tcp_position()
        else:
            cur_tcp = {"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 180}

        from calibration import compute_approach_pose

        approach = compute_approach_pose(center_base, normal_base, cur_tcp)

        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.robot_labels[axis].setText(f"{approach[axis.lower()]:.2f}")

        # 이동 버튼 활성화용으로 타겟 자세 저장
        self.target_pose = approach

        # 이동 / 시퀀스 추가 버튼은 로봇 연결이 되어 있고 타겟이 유효할 때만 활성화
        connected = self.main.robot is not None
        self.btn_move_robot.setEnabled(connected)
        self.btn_add_obj_to_seq.setEnabled(connected)

        # 3D 뷰에 Tool 자세 시각화 (Tool 좌표축 + approach 지점 + 경로선)
        self._render_tcp_visualization()

        # 회전 변화량 계산 (현재 TCP와 비교) → 사용자에게 큰 관절 회전 예상 시 경고
        rot_change = self._compute_rotation_change_deg(cur_tcp, approach)
        rot_part = f", 회전변화 {rot_change:.0f}°" if rot_change is not None else ""
        warn = "  ⚠ 큰 회전 — PTP 권장" if rot_change is not None and rot_change > 60 else ""
        self.main.statusBar().showMessage(
            f"객체 #{idx + 1} 선택: X={approach['x']:.1f}, Y={approach['y']:.1f}, Z={approach['z']:.1f}{rot_part}{warn}"
        )

    def _compute_rotation_change_deg(self, current_tcp, target_tcp):
        """현재 TCP ↔ target TCP 사이의 회전 변화량(axis-angle, °)."""
        try:
            R_cur = tcp_to_homogeneous(current_tcp)[:3, :3]
            R_tgt = tcp_to_homogeneous(target_tcp)[:3, :3]
            R_diff = R_cur.T @ R_tgt
            cos_a = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_a)))
        except Exception:
            return None

    def _render_tcp_visualization(self):
        """
        선택된 객체 바로 위에 그리퍼 접근 자세를 시각화:
          - Tool 좌표축 (X 빨강, Y 초록, Z 파랑) → 그리퍼가 어느 방향으로 접근할지
          - Approach 지점 (주황 구) + target까지 경로선

        3D 뷰는 **카메라 좌표계**라서 베이스 좌표계의 target_pose를 그대로
        그리면 좌표계가 달라 객체와 동떨어진 곳에 표시된다. 따라서:
          - 위치는 obj["center"] (카메라 좌표계 객체 중심) 사용
          - 회전은 베이스 좌표계의 Tool 자세를 카메라 좌표계로 역변환 후 사용
        """
        plotter = self.view_3d.plotter
        for name in self._tcp_viz_actors:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._tcp_viz_actors.clear()

        if self.target_pose is None or self.selected_idx is None or self.T_calib is None:
            plotter.render()
            return
        obj = next((o for o in self.pick_objects if o["index"] == self.selected_idx), None)
        if obj is None:
            plotter.render()
            return

        # 위치: 객체 중심 (카메라 좌표계 — 3D 뷰의 좌표계와 동일)
        origin = np.array(obj["center"], dtype=np.float32)

        # 회전: 베이스 좌표계의 target 자세 → 카메라 좌표계 자세
        R_target_base = tcp_to_homogeneous(self.target_pose)[:3, :3]
        if self.calib_mode == "eye_to_hand":
            # T_cam2base의 회전 부분 역(전치) = base→cam
            R_in_cam = self.T_calib[:3, :3].T @ R_target_base
        elif self.calib_mode == "eye_in_hand":
            if self.main.robot is None:
                plotter.render()
                return
            cur_tcp = self.main.robot.get_tcp_position()
            T_g2b = tcp_to_homogeneous(cur_tcp)
            # T_target_cam = R_cam2gripper.T @ R_gripper2base.T @ R_target_base
            R_in_cam = self.T_calib[:3, :3].T @ T_g2b[:3, :3].T @ R_target_base
        else:
            plotter.render()
            return

        L = 50.0  # 축 길이 mm
        for axis_idx, color, suffix in [(0, "red", "x"), (1, "green", "y"), (2, "blue", "z")]:
            endpoint = (origin + R_in_cam[:, axis_idx] * L).astype(np.float32)
            line = pv.PolyData(np.array([origin, endpoint], dtype=np.float32))
            line.lines = np.array([2, 0, 1])
            name = f"tcp_axis_{suffix}"
            plotter.add_mesh(
                line, color=color, line_width=6, name=name,
                render_lines_as_tubes=True, pickable=False, reset_camera=False,
            )
            self._tcp_viz_actors.append(name)

        # Approach 지점 = Tool -Z 방향으로 offset 떨어진 곳 (카메라 좌표계 -Z)
        offset = float(self.approach_dist.value()) if self.use_approach.isChecked() else 50.0
        approach_pos = (origin - R_in_cam[:, 2] * offset).astype(np.float32)
        sphere = pv.Sphere(radius=4, center=approach_pos)
        plotter.add_mesh(
            sphere, color="#ffaa00", name="tcp_approach", pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("tcp_approach")

        # Approach → Target 경로선
        path = pv.PolyData(np.array([approach_pos, origin], dtype=np.float32))
        path.lines = np.array([2, 0, 1])
        plotter.add_mesh(
            path, color="#ffaa00", line_width=3, name="tcp_path",
            render_lines_as_tubes=True, pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("tcp_path")

        plotter.render()

    # ============================================================
    # 로봇 이동 / 비상정지 제어
    # ============================================================

    def _execute_move(self):
        """선택된 객체의 위치로 로봇 이동 (큐에 모션 추가)"""
        if self.target_pose is None:
            QMessageBox.warning(self, "오류", "먼저 객체를 선택하세요")
            return

        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return

        p = self.target_pose

        # 안전 검증: Z 한계
        if not self._validate_z(p["z"]):
            return

        mode_text = self.move_mode_combo.currentText()
        is_lin = mode_text.startswith("LIN")
        # AUT 모드면 50% 상한 적용
        speed = self._effective_speed(self.speed_spin.value())
        use_approach = self.use_approach.isChecked()
        offset = self.approach_dist.value()

        # 접근/철수 사용 시 위치 미리 계산 (확인 다이얼로그에 표시용)
        approach_xyz = None
        if use_approach:
            approach_xyz = self._compute_approach_position(p, offset)
            # approach 지점 Z도 안전 한계 검증 (Tool +Z가 옆/위면 바닥 아래로 갈 수 있음)
            if not self._validate_z(approach_xyz[2]):
                return

        # 안전 확인 다이얼로그
        if use_approach:
            msg = (
                f"⚠ 로봇 이동 확인 (접근/철수 모드, 큐에 3개 모션 추가)\n\n"
                f"방식: {'LIN (직선)' if is_lin else 'PTP (최단)'}\n"
                f"속도: {speed}%\n"
                f"접근 거리: {offset}mm (법선 바깥 방향)\n\n"
                f"[1] Approach (위로):\n"
                f"  X: {approach_xyz[0]:.2f}, Y: {approach_xyz[1]:.2f}, Z: {approach_xyz[2]:.2f}\n\n"
                f"[2] Target (정밀 접근):\n"
                f"  X: {p['x']:.2f}, Y: {p['y']:.2f}, Z: {p['z']:.2f}\n"
                f"  A: {p['a']:.2f}, B: {p['b']:.2f}, C: {p['c']:.2f}\n\n"
                f"[3] Retract (다시 위로):\n"
                f"  X: {approach_xyz[0]:.2f}, Y: {approach_xyz[1]:.2f}, Z: {approach_xyz[2]:.2f}\n\n"
                f"⚠ T1 모드 - 데드맨+시작 버튼 잡고 있어야 이동\n"
                f"⚠ 비상시 Space 또는 비상정지 버튼\n\n"
                f"진행하시겠습니까?"
            )
        else:
            msg = (
                f"⚠ 로봇 이동 확인 (단일 모션)\n\n"
                f"방식: {'LIN (직선)' if is_lin else 'PTP (최단)'}\n"
                f"속도: {speed}%\n\n"
                f"목표 위치:\n"
                f"  X: {p['x']:.2f} mm\n"
                f"  Y: {p['y']:.2f} mm\n"
                f"  Z: {p['z']:.2f} mm\n"
                f"  A: {p['a']:.2f} °\n"
                f"  B: {p['b']:.2f} °\n"
                f"  C: {p['c']:.2f} °\n\n"
                f"⚠ T1 모드 - 데드맨+시작 버튼 잡고 있어야 이동\n\n"
                f"진행하시겠습니까?"
            )

        ret = QMessageBox.question(self, "로봇 이동 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        try:
            # 속도 설정
            self.main.robot.set_speed(speed)

            slots_added = []

            def add_motion(x, y, z, a, b, c):
                if is_lin:
                    return self.main.robot.add_move_lin(x, y, z, a, b, c)
                else:
                    return self.main.robot.add_move_ptp(x, y, z, a, b, c)

            if use_approach:
                # 1) Approach: 법선 바깥에서 같은 자세
                ax, ay, az = approach_xyz
                slot1 = add_motion(ax, ay, az, p["a"], p["b"], p["c"])
                if slot1 is None:
                    QMessageBox.critical(self, "오류", "Approach 모션 추가 실패")
                    return
                slots_added.append(slot1)

                # 2) Target: 정밀 접근 (LIN 강제 - 안전)
                slot2 = self.main.robot.add_move_lin(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])
                if slot2 is None:
                    QMessageBox.critical(self, "오류", "Target 모션 추가 실패")
                    return
                slots_added.append(slot2)

                # 3) Retract: 다시 Approach 위치로 (LIN 강제 - 안전)
                slot3 = self.main.robot.add_move_lin(ax, ay, az, p["a"], p["b"], p["c"])
                if slot3 is None:
                    QMessageBox.critical(self, "오류", "Retract 모션 추가 실패")
                    return
                slots_added.append(slot3)

                self.main.statusBar().showMessage(f"✅ 접근/철수 3단계 큐 추가 (slots={slots_added}). SmartPAD에서 데드맨+시작 버튼으로 진행")
                logger.info(f"이동 (Approach/Target/Retract) slots={slots_added}, target={p}, offset={offset}")
            else:
                # 단일 모션
                slot = add_motion(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])
                if slot is None:
                    QMessageBox.critical(self, "오류", "큐에 모션 추가 실패")
                    return
                slots_added.append(slot)
                self.main.statusBar().showMessage(f"✅ 모션 큐 추가 (slot={slot}). SmartPAD에서 데드맨+시작 버튼으로 진행")
                logger.info(f"이동 명령: {'LIN' if is_lin else 'PTP'}, slot={slot}, target={p}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이동 명령 실패:\n{e}")
            logger.error(f"이동 명령 오류: {e}")

    # ============================================================
    # AUT 모드 안전 기능
    # ============================================================

    def _refresh_mode_display(self):
        """현재 로봇 모드를 라벨에 표시 (2초마다 호출)"""
        if self.main.robot is None:
            self._current_mode = "?"
            self.mode_label.setText("모드: 미연결")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; " "background-color: #BDBDBD; color: white; border-radius: 3px;")
            return
        try:
            m = self.main.robot.read_variable("$MODE_OP")
            if m:
                self._current_mode = normalize_robot_mode(m)
        except Exception:
            return

        if is_auto_mode(self._current_mode):
            # AUT/EXT는 위험 → 빨간색 강조
            self.mode_label.setText(f"⚠ {self._current_mode} (자동 운용)")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; " "background-color: #D32F2F; color: white; border-radius: 3px;")
        elif "T1" in self._current_mode or "T2" in self._current_mode:
            self.mode_label.setText(f"{self._current_mode} (수동)")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; " "background-color: #2E7D32; color: white; border-radius: 3px;")
        else:
            self.mode_label.setText(f"모드: {self._current_mode}")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; " "background-color: #757575; color: white; border-radius: 3px;")
