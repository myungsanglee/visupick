"""
Hand-Eye Calibration GUI
PySide6 기반 데이터 수집 및 캘리브레이션 검증 프로그램
"""

import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

from kuka_robot import KUKARobot
from camera_factory import create_camera, list_available_camera_names, get_settings_file_filter
from bin_picking_tab import BinPickingTab
from cad_matching_tab import CADMatchingTab
from calibration import (
    compute_hand_eye,
    save_calibration_result,
    tcp_to_homogeneous,
    estimate_normal_at_pixel,
    compute_approach_pose,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ImageViewerMixin:
    """이미지 표시용 공통 로직 (QLabel에 리사이즈 가능한 이미지 표시)"""

    def _init_image_viewer(self):
        self._display_rgb = None
        self.image_label = QLabel("이미지가 여기에 표시됩니다")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2a2a2a; color: #888;")
        self.image_label.setMinimumSize(640, 480)

    def _display_image(self, image: np.ndarray):
        self._display_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._refresh_image()

    def _refresh_image(self):
        if self._display_rgb is None:
            return
        rgb = self._display_rgb
        h, w = rgb.shape[:2]
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if label_w <= 0 or label_h <= 0:
            return
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h))
        qimage = QImage(rgb_resized.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))


class DataCollectionTab(QWidget, ImageViewerMixin):
    """탭 1: 데이터 수집 + 캘리브레이션 실행"""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        self.pose_count = 0
        self.current_frame = None
        self.current_image = None
        self.current_tcp = None
        self.session_dir = None

        self._init_image_viewer()
        self._init_ui()
        # 세션 폴더는 사용자가 명시적으로 "새 세션 시작" 또는
        # "기존 세션 불러오기" 누를 때만 만든다 (자동 생성 X)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 세션 폴더 표시 + 로드 + 새 세션
        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("세션:"))
        self.session_label = QLabel("(세션 미선택 - 새 세션 시작 또는 기존 세션 불러오기)")
        self.session_label.setStyleSheet("font-family: monospace; color: #777;")
        session_row.addWidget(self.session_label)
        session_row.addStretch()
        self.btn_load_session = QPushButton("기존 세션 불러오기")
        self.btn_load_session.clicked.connect(self._load_session)
        session_row.addWidget(self.btn_load_session)
        self.btn_new_session = QPushButton("새 세션 시작")
        self.btn_new_session.clicked.connect(self._new_session)
        session_row.addWidget(self.btn_new_session)
        layout.addLayout(session_row)

        splitter = QSplitter(Qt.Horizontal)

        # 왼쪽: 이미지
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label, stretch=1)
        self.checkerboard_status = QLabel("")
        self.checkerboard_status.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.checkerboard_status)
        splitter.addWidget(image_widget)

        # 오른쪽: 정보
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # TCP 정보
        tcp_group = QGroupBox("현재 TCP 위치")
        tcp_layout = QVBoxLayout(tcp_group)
        self.tcp_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            label = QLabel("---")
            label.setStyleSheet("font-family: monospace; font-size: 14px;")
            self.tcp_labels[axis] = label
            row.addWidget(label)
            row.addStretch()
            tcp_layout.addLayout(row)
        info_layout.addWidget(tcp_group)

        # 캘리브레이션 설정
        cal_group = QGroupBox("캘리브레이션 설정")
        cal_layout = QVBoxLayout(cal_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("모드:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Eye-to-Hand", "Eye-in-Hand"])
        mode_row.addWidget(self.mode_combo)
        mode_row.addSpacing(15)
        mode_row.addWidget(QLabel("포즈 추정:"))
        self.pose_method_combo = QComboBox()
        self.pose_method_combo.addItems([
            "auto (포인트클라우드 우선, 없으면 PnP)",
            "compare (두 방법 비교)",
            "pointcloud (3D 직접 매칭 강제)",
            "pnp (solvePnP 강제 — RealSense 같은 저정밀 깊이용)",
        ])
        self.pose_method_combo.setToolTip(
            "체커보드 자세 추정 방식.\n"
            "auto: Zivid 같은 정밀 깊이 카메라에 적합.\n"
            "compare: 두 방법 모두 계산해 일관성 metric 비교 (진단용, 시간 2배).\n"
            "pointcloud/pnp: 강제 — 디버그/실험용."
        )
        mode_row.addWidget(self.pose_method_combo)
        cal_layout.addLayout(mode_row)

        board_row = QHBoxLayout()
        board_row.addWidget(QLabel("체커보드 크기:"))
        self.board_w = QSpinBox()
        self.board_w.setRange(2, 20)
        self.board_w.setValue(8)
        board_row.addWidget(self.board_w)
        board_row.addWidget(QLabel("x"))
        self.board_h = QSpinBox()
        self.board_h.setRange(2, 20)
        self.board_h.setValue(6)
        board_row.addWidget(self.board_h)
        cal_layout.addLayout(board_row)

        square_row = QHBoxLayout()
        square_row.addWidget(QLabel("칸 크기 (mm):"))
        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(1.0, 200.0)
        self.square_size.setValue(25.0)
        self.square_size.setDecimals(1)
        square_row.addWidget(self.square_size)
        cal_layout.addLayout(square_row)

        info_layout.addWidget(cal_group)

        # 포즈 테이블
        pose_group = QGroupBox("수집된 포즈")
        pose_layout = QVBoxLayout(pose_group)
        self.pose_table = QTableWidget(0, 7)
        self.pose_table.setHorizontalHeaderLabels(["#", "X", "Y", "Z", "A", "B", "C"])
        self.pose_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pose_table.setEditTriggers(QTableWidget.NoEditTriggers)
        pose_layout.addWidget(self.pose_table)

        self.pose_count_label = QLabel("수집된 포즈: 0")
        self.pose_count_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        pose_layout.addWidget(self.pose_count_label)
        info_layout.addWidget(pose_group)

        splitter.addWidget(info_widget)
        splitter.setSizes([850, 350])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 액션 버튼
        btn_layout = QHBoxLayout()

        self.btn_capture = QPushButton("캡처 (C)")
        self.btn_capture.setMinimumHeight(50)
        self.btn_capture.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_capture.clicked.connect(self._capture)
        btn_layout.addWidget(self.btn_capture)

        self.btn_save = QPushButton("저장 (S)")
        self.btn_save.setMinimumHeight(50)
        self.btn_save.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_save.clicked.connect(self._save_pose)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)

        # 탭 활성 시에만 동작하는 단축키
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.setContext(Qt.WidgetWithChildrenShortcut)
        sc_c.activated.connect(self._capture)
        sc_s = QShortcut(QKeySequence("S"), self)
        sc_s.setContext(Qt.WidgetWithChildrenShortcut)
        sc_s.activated.connect(self._save_pose)

        self.btn_calibrate = QPushButton("캘리브레이션 실행")
        self.btn_calibrate.setMinimumHeight(50)
        self.btn_calibrate.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_calibrate.clicked.connect(self._run_calibration)
        btn_layout.addWidget(self.btn_calibrate)

        layout.addLayout(btn_layout)

    def _init_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path("data") / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_label.setText(str(self.session_dir))
        logger.info(f"세션 디렉토리: {self.session_dir}")

    def _new_session(self):
        """새 세션 시작 (테이블 초기화, 디렉토리 새로 생성)"""
        self.pose_count = 0
        self.pose_table.setRowCount(0)
        self.pose_count_label.setText("수집된 포즈: 0")
        self.current_frame = None
        self.current_image = None
        self.btn_save.setEnabled(False)
        self._init_session()
        self.main.statusBar().showMessage("새 세션 시작")

    def _load_session(self):
        """기존 세션 폴더 로드 (pose_*/tcp.json 파일 스캔)"""
        path = QFileDialog.getExistingDirectory(
            self, "세션 폴더 선택", "data",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        session_path = Path(path)
        pose_dirs = sorted(session_path.glob("pose_*"))

        if not pose_dirs:
            QMessageBox.warning(self, "오류", "선택한 폴더에 pose_* 디렉토리가 없습니다")
            return

        # 테이블 초기화
        self.pose_table.setRowCount(0)
        self.session_dir = session_path
        self.session_label.setText(str(self.session_dir))

        loaded = 0
        for pose_dir in pose_dirs:
            tcp_file = pose_dir / "tcp.json"
            if not tcp_file.exists():
                continue
            try:
                with open(tcp_file) as f:
                    tcp_data = json.load(f)
                # 포즈 번호는 디렉토리명에서 추출 (pose_001 → 1)
                try:
                    pose_num = int(pose_dir.name.split("_")[-1])
                except ValueError:
                    pose_num = loaded + 1

                row = self.pose_table.rowCount()
                self.pose_table.insertRow(row)
                self.pose_table.setItem(row, 0, QTableWidgetItem(str(pose_num)))
                for i, axis in enumerate(["x", "y", "z", "a", "b", "c"]):
                    val = tcp_data.get(axis, 0)
                    self.pose_table.setItem(row, i + 1, QTableWidgetItem(f"{val:.2f}"))
                loaded += 1
            except Exception as e:
                logger.warning(f"포즈 로드 실패 ({pose_dir.name}): {e}")

        # pose_count는 가장 큰 번호로 설정 (다음 저장 시 이어서)
        max_num = 0
        for pose_dir in pose_dirs:
            try:
                num = int(pose_dir.name.split("_")[-1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
        self.pose_count = max_num

        self.pose_count_label.setText(f"수집된 포즈: {loaded}")
        self.main.statusBar().showMessage(
            f"세션 로드 완료: {loaded}개 포즈 ({session_path.name})"
        )
        logger.info(f"세션 로드: {session_path}, {loaded}개 포즈")

    def _update_tcp(self):
        if not self.main.robot:
            return
        tcp = self.main.robot.get_tcp_position()
        if tcp:
            self.current_tcp = tcp
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                val = tcp.get(axis.lower(), 0)
                self.tcp_labels[axis].setText(f"{val:.2f}")

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

        self.current_frame = frame
        image = self.main.camera.frame_to_2d_image(frame)
        if image is None:
            self.main.statusBar().showMessage("이미지 추출 실패")
            return
        self.current_image = image

        self._update_tcp()

        board_size = (self.board_w.value(), self.board_h.value())
        found, corners, overlay = self.main.camera.detect_checkerboard(image, board_size)

        if found:
            self.checkerboard_status.setText(f"체커보드 검출 성공 ({len(corners)} 코너)")
            self.checkerboard_status.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            self.btn_save.setEnabled(True)
            display_image = overlay
        else:
            self.checkerboard_status.setText("체커보드 검출 실패")
            self.checkerboard_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.btn_save.setEnabled(False)
            display_image = image

        self._display_image(display_image)
        self.main.statusBar().showMessage("캡처 완료")

    def _save_pose(self):
        if self.session_dir is None:
            QMessageBox.warning(self, "오류", "세션이 선택되지 않았습니다.\n'새 세션 시작' 또는 '기존 세션 불러오기' 버튼을 먼저 누르세요.")
            return
        if self.current_tcp is None or self.current_frame is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 수행하세요")
            return

        self.pose_count += 1
        pose_dir = self.session_dir / f"pose_{self.pose_count:03d}"
        pose_dir.mkdir(parents=True, exist_ok=True)

        tcp_data = {k: v for k, v in self.current_tcp.items() if k in ("x", "y", "z", "a", "b", "c")}
        with open(pose_dir / "tcp.json", "w") as f:
            json.dump(tcp_data, f, indent=2)

        cv2.imwrite(str(pose_dir / "image.png"), self.current_image)

        # 카메라 종류별 native 포인트 클라우드 저장 (Zivid: .zdf, RealSense: .ply)
        pc_ext = ".zdf" if "Zivid" in type(self.main.camera).__name__ else ".ply"
        self.main.camera.save_point_cloud(self.current_frame, str(pose_dir / f"pointcloud{pc_ext}"))
        xyz = self.main.camera.frame_to_point_cloud(self.current_frame)
        if xyz is not None:
            np.save(str(pose_dir / "pointcloud_xyz.npy"), xyz)

        intrinsics_file = self.session_dir / "intrinsics.json"
        if not intrinsics_file.exists():
            self._save_intrinsics(intrinsics_file)

        row = self.pose_table.rowCount()
        self.pose_table.insertRow(row)
        self.pose_table.setItem(row, 0, QTableWidgetItem(str(self.pose_count)))
        for i, axis in enumerate(["x", "y", "z", "a", "b", "c"]):
            val = tcp_data.get(axis, 0)
            self.pose_table.setItem(row, i + 1, QTableWidgetItem(f"{val:.2f}"))

        self.pose_count_label.setText(f"수집된 포즈: {self.pose_count}")
        self.btn_save.setEnabled(False)
        self.main.statusBar().showMessage(f"포즈 {self.pose_count} 저장 완료: {pose_dir}")
        logger.info(f"포즈 {self.pose_count} 저장: {pose_dir}")

    def _save_intrinsics(self, path: Path):
        data = self.main.camera.get_intrinsics()
        if data:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"카메라 내부 파라미터 저장: {path}")

    def _run_calibration(self):
        if self.session_dir is None:
            QMessageBox.warning(self, "오류", "세션이 선택되지 않았습니다.\n'새 세션 시작' 또는 '기존 세션 불러오기' 버튼을 먼저 누르세요.")
            return
        if self.pose_count < 3:
            QMessageBox.warning(self, "오류", f"최소 3개의 포즈가 필요합니다 (현재: {self.pose_count})")
            return

        mode_text = self.mode_combo.currentText()
        mode = "eye_to_hand" if mode_text == "Eye-to-Hand" else "eye_in_hand"
        board_size = (self.board_w.value(), self.board_h.value())
        square_size = self.square_size.value()

        # 포즈 추정 방식 결정 (콤보 첫 단어가 키)
        method_text = self.pose_method_combo.currentText()
        method_key = method_text.split(" ", 1)[0]   # "auto" / "compare" / "pointcloud" / "pnp"

        self.main.statusBar().showMessage("캘리브레이션 계산 중...")
        QApplication.processEvents()

        if method_key == "compare":
            # 두 방법으로 각각 풀어 일관성 metric 비교
            res_pc = compute_hand_eye(
                str(self.session_dir), board_size=board_size, square_size=square_size,
                mode=mode, pose_method="pointcloud", return_metric=True,
            )
            res_pnp = compute_hand_eye(
                str(self.session_dir), board_size=board_size, square_size=square_size,
                mode=mode, pose_method="pnp", return_metric=True,
            )
            self._show_compare_result(res_pc, res_pnp, mode, mode_text)
        else:
            T = compute_hand_eye(
                str(self.session_dir), board_size=board_size, square_size=square_size,
                mode=mode, pose_method=method_key,
            )
            if T is not None:
                result_path = self.session_dir / "calibration_result.json"
                save_calibration_result(T, str(result_path), mode=mode)
                msg = "캘리브레이션 성공!\n\n"
                msg += f"모드: {mode_text}\n"
                msg += f"포즈 추정: {method_key}\n"
                msg += f"사용 포즈: {self.pose_count}\n\n"
                msg += f"변환 행렬:\n{np.array2string(T, precision=4, suppress_small=True)}\n\n"
                msg += f"결과 저장: {result_path}"
                QMessageBox.information(self, "캘리브레이션 결과", msg)
                self.main.statusBar().showMessage("캘리브레이션 완료")
            else:
                QMessageBox.critical(self, "오류", "캘리브레이션 실패. 로그를 확인하세요.")
                self.main.statusBar().showMessage("캘리브레이션 실패")

    def _show_compare_result(self, res_pc, res_pnp, mode, mode_text):
        """두 방법(3D direct, solvePnP) 결과 비교 다이얼로그 + 더 일관성 좋은 쪽 자동 저장."""
        lines = [f"포즈 추정 방식 비교 — 모드: {mode_text}\n"]

        def fmt(name, res):
            if res is None:
                return f"[{name}]  실패 (포즈 부족 또는 데이터 누락)"
            return (
                f"[{name}]  일관성 mean = {res['metric_mean']:.3f} mm, "
                f"사용 포즈 {res['n_used']}/{res['n_total']}, alg={res['algorithm']}"
            )

        lines.append(fmt("Pointcloud (3D 직접 매칭)", res_pc))
        lines.append(fmt("PnP        (solvePnP)     ", res_pnp))
        lines.append("")

        # 더 좋은 (metric 작은) 쪽 자동 채택. 한 쪽만 성공하면 그쪽으로.
        candidates = [(name, r) for name, r in [("pointcloud", res_pc), ("pnp", res_pnp)] if r is not None]
        if not candidates:
            lines.append("⛔ 두 방법 모두 실패. 데이터/intrinsics 를 확인하세요.")
            QMessageBox.critical(self, "캘리브레이션 비교 결과", "\n".join(lines))
            self.main.statusBar().showMessage("캘리브레이션 실패 (양쪽 모두)")
            return

        best_name, best_res = min(candidates, key=lambda x: x[1]["metric_mean"])
        T = best_res["T"]
        result_path = self.session_dir / "calibration_result.json"
        save_calibration_result(T, str(result_path), mode=mode)

        lines.append(f"✅ 더 일관성 좋은 결과: '{best_name}' → 채택해 저장")
        lines.append(f"   결과 파일: {result_path}\n")
        lines.append("변환 행렬:")
        lines.append(np.array2string(T, precision=4, suppress_small=True))

        # 두 결과 차이가 작으면 카메라 정밀도가 양쪽 다 충분, 크면 한쪽이 부정확
        if len(candidates) == 2:
            diff = abs(res_pc["metric_mean"] - res_pnp["metric_mean"])
            lines.append("")
            if diff < 0.5:
                lines.append(f"📌 두 방법 차이 = {diff:.3f} mm (작음). 카메라 깊이/intrinsics 모두 신뢰할 만함.")
            elif diff < 2.0:
                lines.append(f"📌 두 방법 차이 = {diff:.3f} mm (중간). 카메라 종류에 따라 한쪽이 더 정밀할 수 있음.")
            else:
                lines.append(f"⚠ 두 방법 차이 = {diff:.3f} mm (큼). 한쪽이 부정확 — "
                             "Zivid 같으면 pointcloud, RealSense 같으면 pnp 가 신뢰도 ↑")

        QMessageBox.information(self, "캘리브레이션 비교 결과", "\n".join(lines))
        self.main.statusBar().showMessage(f"캘리브레이션 완료 (비교: '{best_name}' 채택)")


class VerificationTab(QWidget, ImageViewerMixin):
    """탭 2: 캘리브레이션 검증 (체커보드 0번 코너 → 로봇 타겟 위치 계산)"""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        self.T_calib = None       # 로드된 캘리브레이션 변환 행렬
        self.calib_mode = None    # "eye_to_hand" or "eye_in_hand"
        self.corner_cam = None    # 체커보드 0번 코너의 카메라 좌표 (3D, mm)
        self.normal_cam = None    # 체커보드 평면 법선 (카메라 좌표계)
        self.current_tcp = None   # 현재 로봇 TCP

        self._init_image_viewer()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 상단: 캘리브레이션 파일 + 설정
        top_row = QHBoxLayout()
        self.btn_load_calib = QPushButton("캘리브레이션 파일 (JSON)")
        self.btn_load_calib.clicked.connect(self._load_calibration)
        top_row.addWidget(self.btn_load_calib)
        self.calib_label = QLabel("미로드")
        top_row.addWidget(self.calib_label)
        self.calib_mode_label = QLabel("")
        self.calib_mode_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        top_row.addWidget(self.calib_mode_label)
        top_row.addSpacing(20)

        top_row.addWidget(QLabel("체커보드:"))
        self.board_w = QSpinBox()
        self.board_w.setRange(2, 20)
        self.board_w.setValue(8)
        self.board_w.setFixedWidth(50)
        top_row.addWidget(self.board_w)
        top_row.addWidget(QLabel("x"))
        self.board_h = QSpinBox()
        self.board_h.setRange(2, 20)
        self.board_h.setValue(6)
        self.board_h.setFixedWidth(50)
        top_row.addWidget(self.board_h)

        top_row.addStretch()
        layout.addLayout(top_row)

        splitter = QSplitter(Qt.Horizontal)

        # 왼쪽: 이미지
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label, stretch=1)
        self.checkerboard_status = QLabel("")
        self.checkerboard_status.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.checkerboard_status)
        splitter.addWidget(image_widget)

        # 오른쪽: 정보 (현재 TCP + 0번 코너 카메라 좌표 + 타겟 TCP)
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # 현재 TCP
        cur_tcp_group = QGroupBox("현재 로봇 TCP")
        cur_tcp_layout = QVBoxLayout(cur_tcp_group)
        self.cur_tcp_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 14px;")
            self.cur_tcp_labels[axis] = lab
            row.addWidget(lab)
            row.addStretch()
            cur_tcp_layout.addLayout(row)
        info_layout.addWidget(cur_tcp_group)

        # 0번 코너 카메라 좌표
        corner_group = QGroupBox("체커보드 0번 코너 (카메라 좌표계)")
        corner_layout = QVBoxLayout(corner_group)
        self.corner_labels = {}
        for axis in ["X", "Y", "Z"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 14px;")
            self.corner_labels[axis] = lab
            row.addWidget(lab)
            row.addWidget(QLabel("mm"))
            row.addStretch()
            corner_layout.addLayout(row)
        info_layout.addWidget(corner_group)

        # 체커보드 평면 법선 (카메라 좌표계)
        normal_group = QGroupBox("체커보드 법선 (카메라 좌표계)")
        normal_layout = QVBoxLayout(normal_group)
        self.normal_labels = {}
        for axis in ["X", "Y", "Z"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"n{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 14px;")
            self.normal_labels[axis] = lab
            row.addWidget(lab)
            row.addStretch()
            normal_layout.addLayout(row)
        info_layout.addWidget(normal_group)

        # 타겟 TCP (회전 현재 유지)
        target_group = QGroupBox("로봇 타겟 위치 (회전은 현재 TCP 유지)")
        target_layout = QVBoxLayout(target_group)
        self.target_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 16px; font-weight: bold; color: #0066cc;")
            self.target_labels[axis] = lab
            row.addWidget(lab)
            row.addStretch()
            target_layout.addLayout(row)
        info_layout.addWidget(target_group)

        # 수직 접근 TCP (법선 방향)
        approach_group = QGroupBox("수직 접근 자세 (법선 방향)")
        approach_layout = QVBoxLayout(approach_group)
        self.approach_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 16px; font-weight: bold; color: #cc6600;")
            self.approach_labels[axis] = lab
            row.addWidget(lab)
            row.addStretch()
            approach_layout.addLayout(row)
        info_layout.addWidget(approach_group)

        info_layout.addStretch()
        splitter.addWidget(info_widget)
        splitter.setSizes([850, 350])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 액션 버튼
        btn_layout = QHBoxLayout()

        self.btn_capture = QPushButton("캡처 (C)")
        self.btn_capture.setMinimumHeight(50)
        self.btn_capture.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_capture.clicked.connect(self._capture)
        btn_layout.addWidget(self.btn_capture)

        self.btn_compute = QPushButton("타겟 위치 계산")
        self.btn_compute.setMinimumHeight(50)
        self.btn_compute.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_compute.clicked.connect(self._compute_target)
        self.btn_compute.setEnabled(False)
        btn_layout.addWidget(self.btn_compute)

        layout.addLayout(btn_layout)

        # 탭 활성 시에만 동작하는 단축키
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.setContext(Qt.WidgetWithChildrenShortcut)
        sc_c.activated.connect(self._capture)

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
            self.calib_label.setText(Path(path).name)
            self.calib_mode_label.setText(f"[{self.calib_mode}]")
            self.main.statusBar().showMessage(f"캘리브레이션 로드: {Path(path).name} ({self.calib_mode})")
            logger.info(f"캘리브레이션 행렬:\n{self.T_calib}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"캘리브레이션 로드 실패:\n{e}")

    def _update_tcp(self):
        if not self.main.robot:
            return
        tcp = self.main.robot.get_tcp_position()
        if tcp:
            self.current_tcp = tcp
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                val = tcp.get(axis.lower(), 0)
                self.cur_tcp_labels[axis].setText(f"{val:.2f}")

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

        self._update_tcp()

        board_size = (self.board_w.value(), self.board_h.value())
        found, corners, overlay = self.main.camera.detect_checkerboard(image, board_size)

        if not found:
            self.checkerboard_status.setText("체커보드 검출 실패")
            self.checkerboard_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.btn_compute.setEnabled(False)
            self._display_image(image)
            return

        # 0번 코너 픽셀 좌표
        corner0_px = corners[0, 0]
        px, py = int(round(corner0_px[0])), int(round(corner0_px[1]))

        h, w = xyz.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            self.checkerboard_status.setText("0번 코너가 이미지 영역을 벗어남")
            self.checkerboard_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self._display_image(overlay)
            return

        point_3d = xyz[py, px]
        if np.any(np.isnan(point_3d)):
            self.checkerboard_status.setText("0번 코너의 3D 좌표가 유효하지 않습니다 (NaN)")
            self.checkerboard_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self._display_image(overlay)
            return

        self.corner_cam = point_3d
        for axis, val in zip(["X", "Y", "Z"], point_3d):
            self.corner_labels[axis].setText(f"{val:.2f}")

        # 0번 코너 주변 XYZ 포인트로 국소 평면 피팅 → 법선 (카메라 좌표계)
        # Zivid normals는 fallback으로 전달
        self.normal_cam = estimate_normal_at_pixel(
            xyz, px, py, patch_radius=15, normals=normals,
        )

        if self.normal_cam is not None:
            for axis, val in zip(["X", "Y", "Z"], self.normal_cam):
                self.normal_labels[axis].setText(f"{val:+.4f}")
        else:
            for axis in ["X", "Y", "Z"]:
                self.normal_labels[axis].setText("---")

        # 0번 코너 강조 (빨간 원)
        cv2.circle(overlay, (px, py), 15, (0, 0, 255), 3)
        cv2.putText(overlay, "0", (px + 20, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        self.checkerboard_status.setText(f"체커보드 검출 성공 ({len(corners)} 코너, 0번 코너 표시됨)")
        self.checkerboard_status.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        self._display_image(overlay)
        self.btn_compute.setEnabled(True)
        self.main.statusBar().showMessage("캡처 완료 - 타겟 위치 계산 가능")

    def _compute_target(self):
        if self.T_calib is None:
            QMessageBox.warning(self, "오류", "캘리브레이션 파일을 먼저 로드하세요")
            return
        if self.corner_cam is None:
            QMessageBox.warning(self, "오류", "먼저 캡처하여 0번 코너를 검출하세요")
            return
        if self.current_tcp is None:
            QMessageBox.warning(self, "오류", "로봇 TCP 값이 없습니다 (로봇 연결 확인)")
            return

        # 코너를 동차 좌표로
        point_cam = np.array([self.corner_cam[0], self.corner_cam[1], self.corner_cam[2], 1.0])

        # 법선 벡터 (없으면 0벡터)
        normal_cam = self.normal_cam if self.normal_cam is not None else None

        if self.calib_mode == "eye_to_hand":
            # T_cam2base: 카메라 → base
            point_base = (self.T_calib @ point_cam)[:3]
            # 법선은 방향이므로 회전 부분만 적용
            if normal_cam is not None:
                normal_base = self.T_calib[:3, :3] @ normal_cam
            else:
                normal_base = None
        elif self.calib_mode == "eye_in_hand":
            # T_cam2gripper @ point_cam = point_gripper, 그 다음 gripper→base
            T_gripper2base = tcp_to_homogeneous(self.current_tcp)
            point_gripper_h = self.T_calib @ point_cam
            point_base = (T_gripper2base @ point_gripper_h)[:3]
            if normal_cam is not None:
                # 카메라 → gripper → base (회전만)
                normal_gripper = self.T_calib[:3, :3] @ normal_cam
                normal_base = T_gripper2base[:3, :3] @ normal_gripper
            else:
                normal_base = None
        else:
            QMessageBox.critical(self, "오류", f"알 수 없는 모드: {self.calib_mode}")
            return

        # === 타겟 위치 (회전은 현재 TCP 유지) ===
        target = {
            "x": float(point_base[0]),
            "y": float(point_base[1]),
            "z": float(point_base[2]),
            "a": self.current_tcp["a"],
            "b": self.current_tcp["b"],
            "c": self.current_tcp["c"],
        }
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.target_labels[axis].setText(f"{target[axis.lower()]:.2f}")

        # === 수직 접근 자세 (법선 방향) ===
        if normal_base is not None:
            approach = compute_approach_pose(point_base, normal_base, self.current_tcp)
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                self.approach_labels[axis].setText(f"{approach[axis.lower()]:.2f}")
            logger.info(f"계산된 수직 접근 자세: {approach}")
        else:
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                self.approach_labels[axis].setText("---")

        self.main.statusBar().showMessage(
            f"타겟 위치: X={target['x']:.2f}, Y={target['y']:.2f}, Z={target['z']:.2f}"
        )
        logger.info(f"계산된 타겟 위치: {target}")


class HandEyeCalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand-Eye Calibration")
        self.setMinimumSize(1200, 800)

        self.robot = None
        self.camera = None
        self.home_pose = None  # 로봇 연결 시 저장되는 홈 위치

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 상단: 공유 연결 섹션
        conn_layout = QHBoxLayout()
        conn_layout.setContentsMargins(0, 0, 0, 0)

        conn_layout.addWidget(QLabel("IP:"))
        self.robot_ip_input = QLineEdit("192.168.20.77")
        self.robot_ip_input.setFixedWidth(120)
        conn_layout.addWidget(self.robot_ip_input)

        conn_layout.addWidget(QLabel("Tool:"))
        self.tool_num_input = QSpinBox()
        self.tool_num_input.setRange(0, 16)
        self.tool_num_input.setValue(1)
        self.tool_num_input.setFixedWidth(50)
        conn_layout.addWidget(self.tool_num_input)

        conn_layout.addWidget(QLabel("Base:"))
        self.base_num_input = QSpinBox()
        self.base_num_input.setRange(0, 16)
        self.base_num_input.setValue(0)
        self.base_num_input.setFixedWidth(50)
        conn_layout.addWidget(self.base_num_input)

        self.btn_connect_robot = QPushButton("로봇 연결")
        self.btn_connect_robot.clicked.connect(self._connect_robot)
        conn_layout.addWidget(self.btn_connect_robot)
        self.robot_status = QLabel("미연결")
        self.robot_status.setStyleSheet("color: red; font-weight: bold;")
        conn_layout.addWidget(self.robot_status)

        conn_layout.addSpacing(15)

        # 카메라 종류 콤보 (Zivid / RealSense 등 — base_camera 인터페이스만 맞으면 추가 가능)
        conn_layout.addWidget(QLabel("카메라:"))
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(list_available_camera_names())
        self.camera_type_combo.setToolTip(
            "카메라 종류 선택. SDK가 설치된 카메라만 연결 가능.\n"
            "Zivid: 정밀(0.1mm). RealSense: 저렴/빠름(~2mm)."
        )
        conn_layout.addWidget(self.camera_type_combo)

        self.btn_connect_camera = QPushButton("카메라 연결")
        self.btn_connect_camera.clicked.connect(self._connect_camera)
        conn_layout.addWidget(self.btn_connect_camera)
        self.camera_status = QLabel("미연결")
        self.camera_status.setStyleSheet("color: red; font-weight: bold;")
        conn_layout.addWidget(self.camera_status)

        conn_layout.addSpacing(15)

        self.btn_load_settings = QPushButton("카메라 설정 파일")
        self.btn_load_settings.clicked.connect(self._load_camera_settings)
        conn_layout.addWidget(self.btn_load_settings)
        self.settings_label = QLabel("미로드")
        conn_layout.addWidget(self.settings_label)

        conn_layout.addStretch()

        conn_widget = QWidget()
        conn_widget.setLayout(conn_layout)
        conn_widget.setFixedHeight(conn_widget.sizeHint().height())
        main_layout.addWidget(conn_widget)

        # 탭
        self.tabs = QTabWidget()
        self.data_tab = DataCollectionTab(self)
        self.verification_tab = VerificationTab(self)
        self.bin_picking_tab = BinPickingTab(self)
        self.cad_matching_tab = CADMatchingTab(self)
        self.tabs.addTab(self.data_tab, "데이터 수집")
        self.tabs.addTab(self.verification_tab, "검증")
        self.tabs.addTab(self.bin_picking_tab, "Bin Picking")
        self.tabs.addTab(self.cad_matching_tab, "CAD 매칭")
        main_layout.addWidget(self.tabs)

        self.statusBar().showMessage("프로그램 시작됨")

    def _connect_robot(self):
        try:
            ip = self.robot_ip_input.text().strip()
            tool = self.tool_num_input.value()
            base = self.base_num_input.value()
            self.robot = KUKARobot(ip, tool_num=tool, base_num=base)
            if self.robot.connect():
                self.robot_status.setText("연결됨")
                self.robot_status.setStyleSheet("color: green; font-weight: bold;")
                self.btn_connect_robot.setEnabled(False)
                self.robot_ip_input.setEnabled(False)
                self.tool_num_input.setEnabled(False)
                self.base_num_input.setEnabled(False)

                # 연결 시점의 TCP를 home으로 저장
                self.home_pose = self.robot.get_tcp_position()
                if self.home_pose:
                    logger.info(
                        f"Home 위치 저장: X={self.home_pose['x']:.2f}, "
                        f"Y={self.home_pose['y']:.2f}, Z={self.home_pose['z']:.2f}"
                    )

                self.statusBar().showMessage(
                    f"로봇 연결 성공 (Tool:{tool}, Base:{base}, Home 저장됨)"
                )
                self.data_tab._update_tcp()
                self.verification_tab._update_tcp()
                # bin picking / CAD 매칭 탭에도 home 저장 알림 (UI 갱신)
                if hasattr(self, "bin_picking_tab"):
                    self.bin_picking_tab._on_robot_connected()
                if hasattr(self, "cad_matching_tab"):
                    self.cad_matching_tab._on_robot_connected()
            else:
                self.robot_status.setText("연결 실패")
                self.robot = None
        except Exception as e:
            QMessageBox.critical(self, "오류", f"로봇 연결 실패:\n{e}")

    def _connect_camera(self):
        """카메라 연결 / 해제 토글.
        연결 후 사용자가 다른 카메라로 바꾸려면 일단 해제 → 콤보 변경 → 재연결."""
        # 이미 연결되어 있으면 해제만 수행
        if self.camera is not None and self.camera.connected:
            try:
                self.camera.disconnect()
            except Exception as e:
                logger.warning(f"카메라 disconnect 중 오류 (무시): {e}")
            self.camera = None
            self.camera_status.setText("미연결")
            self.camera_status.setStyleSheet("color: red; font-weight: bold;")
            self.btn_connect_camera.setText("카메라 연결")
            self.camera_type_combo.setEnabled(True)
            self.settings_label.setText("미로드")
            self.statusBar().showMessage("카메라 연결 해제됨 (다른 카메라 선택 후 재연결 가능)")
            return

        # 연결 시도
        kind = self.camera_type_combo.currentText()
        try:
            self.camera = create_camera(kind)   # factory: 지연 import + 인스턴스화
            if self.camera.connect():
                self.camera_status.setText(f"연결됨 ({kind})")
                self.camera_status.setStyleSheet("color: green; font-weight: bold;")
                self.btn_connect_camera.setText("카메라 연결 해제")
                self.camera_type_combo.setEnabled(False)
                self.statusBar().showMessage(f"{kind} 카메라 연결 성공")
            else:
                self.camera_status.setText(f"{kind} 연결 실패")
                self.camera = None
        except Exception as e:
            QMessageBox.critical(self, "오류", f"카메라 연결 실패:\n{e}")
            self.camera = None

    def _load_camera_settings(self):
        # 카메라 종류별 파일 필터 (Zivid → YML, RealSense → JSON 등).
        # 카메라 연결 전이면 콤보 현재값 기준, 연결 후면 그 카메라의 종류 기준.
        kind = self.camera_type_combo.currentText()
        camera_filter = get_settings_file_filter(kind)
        file_filter = f"{camera_filter};;All Files (*)"

        # DontUseNativeDialog: VTK/PyVista가 OpenGL context를 점유하고 있어 GNOME/KDE
        # native 다이얼로그가 빈 창으로 뜨는 문제 회피 (Linux + PySide6 + PyVista 조합).
        path, _ = QFileDialog.getOpenFileName(
            self, f"카메라 설정 파일 선택 ({kind})",
            str(Path(__file__).parent / "config"),
            file_filter,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path and self.camera:
            if self.camera.load_settings(path):
                self.settings_label.setText(Path(path).name)
                self.statusBar().showMessage(f"카메라 설정 로드: {Path(path).name}")
            else:
                QMessageBox.warning(self, "오류", "카메라 설정 로드 실패")
        elif not self.camera:
            QMessageBox.warning(self, "오류", "카메라를 먼저 연결하세요")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 두 탭의 이미지 모두 리프레시
        if hasattr(self, "data_tab"):
            self.data_tab._refresh_image()
        if hasattr(self, "verification_tab"):
            self.verification_tab._refresh_image()

    def closeEvent(self, event):
        # PyVista QtInteractor와 살아있는 QTimer가 Qt 위젯 destroy 순서와 꼬여
        # segfault를 내는 케이스가 잦다. 명시적으로 먼저 정리.
        for tab_attr in ("bin_picking_tab", "cad_matching_tab"):
            tab = getattr(self, tab_attr, None)
            if tab is None:
                continue
            timer = getattr(tab, "_mode_timer", None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
            for view_attr in ("view_3d", "view_cad", "view_cluster"):
                view = getattr(tab, view_attr, None)
                if view is not None:
                    try:
                        view.plotter.close()
                    except Exception:
                        pass

        if self.robot:
            self.robot.disconnect()
        if self.camera:
            self.camera.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HandEyeCalibrationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
