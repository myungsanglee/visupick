"""
VisionTabMixin — 캘리브레이션을 쓰는 탭 4개(빈 픽킹·CAD 매칭·표면 추적·검증)의 공통 기능.

캘리브레이션 상태는 main(VisuPickApp)의 CalibrationContext 하나가 소유하고
(calibration.py — 왜 한 곳인지 그쪽 docstring 참고), 탭은 이 믹스인의
T_calib/calib_mode **읽기 전용 프로퍼티**로 접근한다. 예전처럼 탭이
`self.T_calib = ...` 로 자기 복사본을 만들면 AttributeError 가 나도록 의도했다
— 상태가 두 곳에 살기 시작하면 다시 어긋난다.

`캘리브레이션 로드` 버튼도 여기 한 벌만 있다: 어느 탭에서 로드하든 main 이
모든 탭에 방송(_broadcast_calibration_loaded)해 라벨이 함께 갱신되고, 탭별
후처리는 _on_calibration_loaded() 훅으로 처리한다.

사용법: class XxxTab(VisionTabMixin, ..., QWidget) 으로 상속하고,
탭의 _init_ui 가 self.calib_label(QLabel) 을 만들어 두면 된다.
"""

import logging

import cv2
import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from calibration import tcp_to_homogeneous

logger = logging.getLogger(__name__)


class VisionTabMixin:
    # ------------------------------------------------------------
    # 캘리브레이션 (읽기는 프로퍼티, 로드는 공용 한 벌)
    # ------------------------------------------------------------

    @property
    def T_calib(self):
        """4×4 cam 변환 (main.calib 소유 — 탭 로컬 복사본을 두지 않는다)."""
        return self.main.calib.T

    @property
    def calib_mode(self):
        return self.main.calib.mode

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "캘리브레이션 파일 선택",
            "data",
            "JSON Files (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            name = self.main.calib.load(path)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"로드 실패:\n{e}")
            return
        self.main.statusBar().showMessage(f"캘리브레이션 로드: {name} ({self.calib_mode}) — 모든 탭에 적용")
        self.main._broadcast_calibration_loaded(name)

    def _update_calib_label(self, name: str):
        """캘리브레이션 라벨 갱신 (방송으로 호출됨). 라벨 구성이 다른 탭은 오버라이드."""
        label = getattr(self, "calib_label", None)
        if label is not None:
            label.setText(f"{name} [{self.calib_mode}]")

    def _on_calibration_loaded(self):
        """캘리브레이션이 (어느 탭에서든) 로드된 뒤 탭별 후처리 훅. 기본은 없음."""

    # ------------------------------------------------------------
    # 2D/3D 뷰 전환 (탭은 _init_ui 에서 self._view_pages 만 선언)
    # ------------------------------------------------------------
    #   self._view_pages = [(버튼, None), (버튼, 3D뷰), ...]  # 스택 인덱스 순서
    # 3D 뷰는 전환 직후 위젯 크기가 확정된 뒤 카메라를 재적용해야 해서
    # (Qt 리사이즈가 끝난 다음) singleShot(0) 으로 지연 호출한다.

    def _switch_view(self, idx: int):
        self.view_stack.setCurrentIndex(idx)
        for i, (btn, view) in enumerate(self._view_pages):
            btn.setChecked(i == idx)
        view = self._view_pages[idx][1]
        if view is not None:
            QTimer.singleShot(0, view.refresh_camera)

    # ------------------------------------------------------------
    # 캡처 골격 (연결 확인 → capture → 2D/XYZ/법선 추출 → 탭별 후처리 훅)
    # ------------------------------------------------------------

    # 법선 맵까지 읽는 탭(빈 픽킹 자세 추정·표면 추적 수직 자세)은 True.
    # CAD 매칭은 정합이 자체적으로 법선을 계산하므로 False.
    CAPTURE_READS_NORMALS = True

    def _capture(self):
        """캡처 공통 골격. 탭별 화면 갱신/상태 리셋은 _on_capture() 훅에서."""
        cam = self.main.camera
        if not cam or not cam.connected:
            QMessageBox.warning(self, "오류", "카메라가 연결되지 않았습니다")
            return
        if not cam.is_capture_ready:
            QMessageBox.warning(self, "오류", "카메라가 캡처 준비되지 않았습니다 (Zivid 는 YML 로드 필요)")
            return

        self.main.statusBar().showMessage("캡처 중...")
        QApplication.processEvents()

        frame = cam.capture()
        if frame is None:
            self.main.statusBar().showMessage("캡처 실패")
            return

        image = cam.frame_to_2d_image(frame)  # BGR
        xyz = cam.frame_to_point_cloud(frame)  # (H, W, 3) mm
        if image is None or xyz is None:
            self.main.statusBar().showMessage("데이터 추출 실패")
            return

        self.current_image = image
        self.current_xyz = xyz
        self.current_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.CAPTURE_READS_NORMALS:
            self.current_normals = cam.frame_to_normals(frame)  # (H, W, 3)

        intr_data = cam.get_intrinsics()
        if intr_data:
            self.current_intrinsics = np.array(intr_data["camera_matrix"])

        self._on_capture(image, xyz)

    def _on_capture(self, image, xyz):
        """캡처 성공 후 탭별 후처리 (뷰 갱신·이전 결과 리셋·상태바). 각 탭이 구현."""
        raise NotImplementedError

    # ------------------------------------------------------------
    # 그리퍼 접근 자세 3D 시각화 (빈 픽킹 / CAD 매칭 공용)
    # ------------------------------------------------------------

    def _tcp_viz_origin(self):
        """시각화 원점(카메라 좌표계, np.float32 3벡터) 또는 None. 탭별 오버라이드:
        빈 픽킹 = 선택 객체 중심, CAD 매칭 = 인스턴스 grasp 점."""
        return None

    def _render_tcp_visualization(self):
        """선택 대상 위에 그리퍼 접근 자세를 시각화 (Tool 좌표축 + Approach 구 + 경로선).

        3D 뷰는 **카메라 좌표계**라서 베이스 좌표계의 target_pose 를 그대로 그리면
        엉뚱한 곳에 표시된다. 그래서 위치는 탭이 주는 카메라 좌표계 원점
        (_tcp_viz_origin)을, 회전은 base 자세를 cam 으로 역변환해 사용한다
        (cam→base 회전의 전치 = base→cam — 분기는 CalibrationContext 소유).
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
        origin = self._tcp_viz_origin()
        if origin is None:
            plotter.render()
            return

        R_target_base = tcp_to_homogeneous(self.target_pose)[:3, :3]
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        T_c2b = self.main.calib.T_cam_to_base(tcp)
        if T_c2b is None:
            plotter.render()
            return
        R_in_cam = T_c2b[:3, :3].T @ R_target_base

        L = 50.0  # 축 길이 mm
        for axis_idx, color, suffix in [(0, "red", "x"), (1, "green", "y"), (2, "blue", "z")]:
            endpoint = (origin + R_in_cam[:, axis_idx] * L).astype(np.float32)
            line = pv.PolyData(np.array([origin, endpoint], dtype=np.float32))
            line.lines = np.array([2, 0, 1])
            name = f"tcp_axis_{suffix}"
            plotter.add_mesh(line, color=color, line_width=6, name=name, render_lines_as_tubes=True, pickable=False, reset_camera=False)
            self._tcp_viz_actors.append(name)

        # Approach 지점 = Tool -Z 방향으로 offset 떨어진 곳 (주황 구 + 경로선)
        offset = float(self.approach_dist.value()) if self.use_approach.isChecked() else 50.0
        approach_pos = (origin - R_in_cam[:, 2] * offset).astype(np.float32)
        plotter.add_mesh(pv.Sphere(radius=4, center=approach_pos), color="#ffaa00", name="tcp_approach", pickable=False, reset_camera=False)
        self._tcp_viz_actors.append("tcp_approach")

        path = pv.PolyData(np.array([approach_pos, origin], dtype=np.float32))
        path.lines = np.array([2, 0, 1])
        plotter.add_mesh(path, color="#ffaa00", line_width=3, name="tcp_path", render_lines_as_tubes=True, pickable=False, reset_camera=False)
        self._tcp_viz_actors.append("tcp_path")

        plotter.render()
