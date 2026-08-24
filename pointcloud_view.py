"""
3D 포인트 클라우드 뷰 (PyVista)
===============================
빈 픽킹 / CAD 매칭 / 표면 추적 탭이 공유하는 3D 뷰 위젯. 원래 bin_picking_tab.py 에
있었으나 세 탭이 공유하므로(그리고 CAD 탭은 scene/CAD/cluster 3개 인스턴스로 사용)
중립 모듈로 분리했다 — 탭이 다른 탭을 import 하던 결합을 제거.
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

logger = logging.getLogger(__name__)


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

    # 8 코너 → 12 모서리 인덱스 (0~3 아랫면, 4~7 윗면, 같은 순서로 대응)
    _BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]

    def show_wire_box(self, corners, name: str, color: str = "yellow", line_width: int = 4):
        """임의 방향(회전 포함) 상자를 8 코너로 받아 12 모서리 와이어프레임으로 표시.

        `show_roi_box` 는 축 정렬(AABB)만 그릴 수 있는데, Bin Box 는 base 좌표계에서
        정의되고 yaw 회전이 있어 카메라 좌표계로 옮기면 비스듬한 상자가 된다.
        그래서 코너를 직접 받아 선분으로 그린다.

        corners: (8, 3) — 0~3 아랫면, 4~7 윗면 (같은 순서로 위아래 대응)
        """
        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape != (8, 3):
            logger.warning(f"show_wire_box: 코너 8개 필요 (받은 shape={pts.shape})")
            return
        lines = []
        for a, b in self._BOX_EDGES:
            lines.extend([2, a, b])  # VTK 선분 포맷: [점 개수, idx0, idx1]
        poly = pv.PolyData(pts)
        poly.lines = np.asarray(lines)
        self.plotter.add_mesh(poly, color=color, line_width=line_width, name=name,
                              pickable=False, render_lines_as_tubes=True, reset_camera=False)

    def remove_named(self, name: str):
        """이름으로 액터 제거 (없으면 무시)."""
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass

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

