"""
Bin Picking 탭
- 2D 이미지 뷰: 캡처, 마우스 드래그 ROI, RF-DETR 객체 탐지 bbox 표시
- 3D 포인트 클라우드 뷰 (PyVista): ROI 박스, 검출 객체 마커, 마우스 클릭 선택
- Hand-eye calibration 적용하여 로봇 base 좌표계로 피킹 위치 변환
"""

import sys
import os
import json
import time
import logging
import numpy as np
import cv2
from typing import Optional, List, Dict

from PySide6.QtCore import Qt, QRect, QPoint, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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
    QScrollArea,
    QLineEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)

import pyvista as pv
from pyvistaqt import QtInteractor
import open3d as o3d

# 객체 검출기(RF-DETR / SAM3)는 object_detector 모듈로 분리됨 (base_camera 처럼).
# 아래 경로 환경변수는 그 모듈의 검출기 생성 시 넘겨준다.
#   RFDETR_DETECTOR_DIR : detector.py(RF-DETR 래퍼)가 있는 디렉터리 (sys.path 추가용).
#   SAM3_MODEL_DIR      : SAM3 repo 가 pip 경로에 없을 때 repo 루트.
RFDETR_DETECTOR_DIR = os.environ.get("RFDETR_DETECTOR_DIR", "/home/robotegra/michael/rf-detr/tmp")
SAM3_MODEL_DIR = os.environ.get("SAM3_MODEL_DIR", "")

# [개발용 디버그] True 로 바꾸면 "여는 방향" 실행 시 객체별 영상처리 단계
# (OBB 크롭 원본 → mask → 침식 → Sobel/밝기 가중치 → 가중치×침식)를 cv2.imshow
# 몽타주 창으로 순서대로 띄운다. 아무 키나 누르면 다음 객체로 넘어간다.
# 실제 계산에는 영향 없음. 환경변수 VISUPICK_OPENING_DEBUG=1 로도 켤 수 있다.
OPENING_DEBUG = os.environ.get("VISUPICK_OPENING_DEBUG", "0") == "1"

from calibration import tcp_to_homogeneous
from kuka_robot import normalize_robot_mode, is_auto_mode
from robot_control_mixin import RobotControlMixin
from vision_tab_mixin import VisionTabMixin
from image_view import DraggableImageLabel
from pointcloud_view import PointCloudView3D
import opening_analysis as oa

logger = logging.getLogger(__name__)


# ============================================================
# Bin Box (작업 볼륨) 설정 다이얼로그
# ============================================================


class BinBoxDialog(QDialog):
    """빈(bin) 상자를 **로봇 base 좌표계**로 정의 — 충돌 방지의 기준.

    ROI 와 Bin Box 는 같은 것이다 — **XY·크기는 2D 뷰 드래그로** 정하고, 이 창은 **수정 전용**:
      · 회전(yaw) — 2D 드래그로는 만들 수 없어 여기서만 조정
      · 높이(z_rim/z_floor) — **로봇 티칭** 권장. 측정 깊이는 빈 안 물체·투명체에 좌우되므로
        그리퍼 끝을 바닥/림에 대고 현재 TCP Z 를 읽는 게 가장 정확하다.
      · 그리퍼 반경·여유 — 충돌 검사(파지 허용 영역)용
    """

    def __init__(self, cfg: Dict, get_current_tcp=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bin Box 설정 — 작업 볼륨 (로봇 base 좌표계)")
        self.cfg = dict(cfg)
        self._get_tcp = get_current_tcp
        self.setMinimumWidth(480)

        root = QVBoxLayout(self)

        hint_top = QLabel(
            "XY·크기는 2D 뷰에서 <b>드래그</b>로 설정합니다. 여기서는 값을 미세 조정하세요 — "
            "특히 <b>회전(yaw)</b>은 드래그로 만들 수 없어 여기서만 조정합니다."
        )
        hint_top.setWordWrap(True)
        hint_top.setStyleSheet("color:#555; padding:4px;")
        root.addWidget(hint_top)

        # --- 위치/크기 ---
        g1 = QGroupBox("① 빈 위치·크기 (base XY)")
        f1 = QFormLayout(g1)
        self.cx = self._mk_spin(-3000, 3000, " mm")
        self.cy = self._mk_spin(-3000, 3000, " mm")
        self.sx = self._mk_spin(1, 3000, " mm")
        self.sy = self._mk_spin(1, 3000, " mm")
        self.yaw = self._mk_spin(-180, 180, " °")
        f1.addRow("중심 X", self.cx)
        f1.addRow("중심 Y", self.cy)
        f1.addRow("가로 (X방향)", self.sx)
        f1.addRow("세로 (Y방향)", self.sy)
        self.yaw.setToolTip("빈이 base 축과 나란하지 않을 때의 회전각 (base Z 축 기준)")
        f1.addRow("회전 yaw", self.yaw)
        root.addWidget(g1)

        # --- 높이 (티칭) ---
        g2 = QGroupBox("② 빈 높이 (base Z) — 로봇 티칭 권장")
        f2 = QFormLayout(g2)
        self.z_rim = self._mk_spin(-2000, 3000, " mm")
        self.z_floor = self._mk_spin(-2000, 3000, " mm")
        self.z_rim.setToolTip("빈 상단(림) 높이 — 이 위로 '진입 안전고'만큼 띄워서 빈을 드나든다")
        self.z_floor.setToolTip("빈 바닥 높이 — 하강 하한 (바닥 여유를 더해 사용)")

        rim_row = QHBoxLayout()
        rim_row.addWidget(self.z_rim)
        self.btn_teach_rim = QPushButton("📍 현재 TCP Z 로")
        self.btn_teach_rim.setToolTip("그리퍼 끝을 빈 상단(림)에 댄 상태에서 누르면 현재 TCP Z 를 넣는다")
        self.btn_teach_rim.clicked.connect(lambda: self._teach(self.z_rim, "림"))
        rim_row.addWidget(self.btn_teach_rim)
        f2.addRow("림(상단) Z", self._wrap(rim_row))

        floor_row = QHBoxLayout()
        floor_row.addWidget(self.z_floor)
        self.btn_teach_floor = QPushButton("📍 현재 TCP Z 로")
        self.btn_teach_floor.setToolTip("그리퍼 끝을 빈 바닥에 댄 상태에서 누르면 현재 TCP Z 를 넣는다")
        self.btn_teach_floor.clicked.connect(lambda: self._teach(self.z_floor, "바닥"))
        floor_row.addWidget(self.btn_teach_floor)
        f2.addRow("바닥 Z", self._wrap(floor_row))
        root.addWidget(g2)

        # --- 충돌 파라미터 ---
        g3 = QGroupBox("③ 그리퍼·여유 (충돌 검사용)")
        f3 = QFormLayout(g3)
        self.gripper_r = self._mk_spin(0, 300, " mm")
        self.gripper_r.setToolTip(
            "그리퍼를 원기둥으로 근사했을 때의 반경 (흡착패드+하우징 최대 반경).\n파지점이 벽에서 이만큼 떨어져 있어야 안 부딪힌다."
        )
        self.wall_margin = self._mk_spin(0, 200, " mm")
        self.wall_margin.setToolTip("벽 추가 안전 여유")
        self.floor_margin = self._mk_spin(0, 200, " mm")
        self.floor_margin.setToolTip("바닥 추가 안전 여유 (하강 하한 = 바닥 Z + 이 값)")
        self.safe_height = self._mk_spin(0, 500, " mm")
        self.safe_height.setToolTip("빈에 드나들 때 림 위로 띄우는 높이 — 이 높이에서만 횡이동한다")
        f3.addRow("그리퍼 반경", self.gripper_r)
        f3.addRow("벽 여유", self.wall_margin)
        f3.addRow("바닥 여유", self.floor_margin)
        f3.addRow("진입 안전고", self.safe_height)
        root.addWidget(g3)

        hint = QLabel("3D 뷰: 노란 상자 = 빈 외곽, 초록 상자 = 파지 허용 영역(벽에서 그리퍼 반경+여유 안쪽)")
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._load(self.cfg)

    @staticmethod
    def _mk_spin(lo, hi, suffix):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(1)
        sp.setSuffix(suffix)
        sp.setFixedWidth(120)
        return sp

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _load(self, c: Dict):
        self.cx.setValue(c["cx"])
        self.cy.setValue(c["cy"])
        self.sx.setValue(c["sx"])
        self.sy.setValue(c["sy"])
        self.yaw.setValue(c["yaw"])
        self.z_rim.setValue(c["z_rim"])
        self.z_floor.setValue(c["z_floor"])
        self.gripper_r.setValue(c["gripper_r"])
        self.wall_margin.setValue(c["wall_margin"])
        self.floor_margin.setValue(c["floor_margin"])
        self.safe_height.setValue(c["safe_height"])

    def _teach(self, spin: QDoubleSpinBox, label: str):
        if self._get_tcp is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되어 있지 않습니다")
            return
        tcp = self._get_tcp()
        if not tcp:
            QMessageBox.warning(self, "오류", "현재 로봇 자세를 읽지 못했습니다")
            return
        spin.setValue(float(tcp["z"]))
        QMessageBox.information(self, "티칭 완료", f"{label} Z = {tcp['z']:.1f} mm 로 설정했습니다.")

    def result_config(self) -> Dict:
        return {
            "cx": self.cx.value(),
            "cy": self.cy.value(),
            "sx": self.sx.value(),
            "sy": self.sy.value(),
            "yaw": self.yaw.value(),
            "z_rim": self.z_rim.value(),
            "z_floor": self.z_floor.value(),
            "gripper_r": self.gripper_r.value(),
            "wall_margin": self.wall_margin.value(),
            "floor_margin": self.floor_margin.value(),
            "safe_height": self.safe_height.value(),
        }


# ============================================================
# Grasp(파지) 설정 다이얼로그
# ============================================================


class GraspConfigDialog(QDialog):
    """검출된 객체를 어떻게 잡을지(파지 전략) 설정. CAD 없이 AI 검출로 잡을 때,
    노이즈 있는 3D 전부를 쓰지 않고 특정 자유도를 고정/규칙화한다 (4-DOF top-down 등).

    cfg 딕셔너리(부모 탭의 self.grasp_config)를 편집해 accept 시 반영한다.
    get_current_tcp: 티칭 버튼이 현재 로봇 TCP {x,y,z,a,b,c} 를 받아올 콜백(없으면 None).
    """

    def __init__(self, cfg: Dict, get_current_tcp=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grasp 설정 — 검출 객체를 어떻게 잡을지")
        self.cfg = dict(cfg)  # 편집용 복사본 (accept 시 원본에 반영)
        self._get_tcp = get_current_tcp
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)

        # --- 위치 ---
        pos_group = QGroupBox("① 파지 위치")
        pos_form = QFormLayout(pos_group)
        self.pos_mode = QComboBox()
        self.pos_mode.addItem("고정 평면에 광선 투영 (깊이 불사용, 투명체 권장)", "plane")
        self.pos_mode.addItem("3D 중심 (기존, 깊이 필요)", "cloud")
        self.pos_mode.setToolTip("plane: 픽셀을 base 평면에 투영해 XY 계산 + Z 고정 → 투명체에 안정.\ncloud: 마스크 3D점 median (기존).")
        pos_form.addRow("위치 방식", self.pos_mode)

        self.xy_source = QComboBox()
        self.xy_source.addItem("OBB 중심", "obb_center")
        self.xy_source.addItem("마스크 중심", "mask_center")
        self.xy_source.addItem("bbox 중심", "bbox_center")
        self.xy_source.setToolTip("plane 방식에서 어느 픽셀을 평면에 투영할지.")
        pos_form.addRow("XY 출처 (평면 방식)", self.xy_source)

        self.z_plane = QDoubleSpinBox()
        self.z_plane.setRange(-2000, 2000)
        self.z_plane.setDecimals(1)
        self.z_plane.setSuffix(" mm")
        self.z_plane.setToolTip("광선을 투영할 base 평면 높이 Z. 보통 케이스 윗면 높이. XY·회전 방향 계산에 사용.")
        pos_form.addRow("작업 평면 Z", self.z_plane)

        self.z_pick = QDoubleSpinBox()
        self.z_pick.setRange(-2000, 2000)
        self.z_pick.setDecimals(1)
        self.z_pick.setSuffix(" mm")
        self.z_pick.setToolTip("그리퍼가 실제로 내려갈 파지 Z (base). 투명체는 깊이 대신 이 고정값을 쓴다.")
        pos_form.addRow("파지 Z (고정)", self.z_pick)
        root.addWidget(pos_group)

        # --- 접근 ---
        app_group = QGroupBox("② 접근 방향 (Tool +Z)")
        app_form = QFormLayout(app_group)
        self.approach = QComboBox()
        self.approach.addItem("수직 top-down (B·C 고정)", "vertical")
        self.approach.addItem("표면 법선 (기존)", "normal")
        self.approach.setToolTip("vertical: 항상 수직으로 접근, B·C 고정 → 투명체에 안정.\nnormal: 표면 법선 방향(기존).")
        app_form.addRow("접근 방식", self.approach)

        self.b_fixed = QDoubleSpinBox()
        self.b_fixed.setRange(-180, 180)
        self.b_fixed.setDecimals(2)
        self.b_fixed.setSuffix(" °")
        app_form.addRow("고정 B", self.b_fixed)
        self.c_fixed = QDoubleSpinBox()
        self.c_fixed.setRange(-180, 180)
        self.c_fixed.setDecimals(2)
        self.c_fixed.setSuffix(" °")
        app_form.addRow("고정 C", self.c_fixed)
        root.addWidget(app_group)

        # --- 회전 ---
        yaw_group = QGroupBox("③ 회전 (수직축 A / 그리퍼 X축 방향)")
        yaw_form = QFormLayout(yaw_group)
        self.yaw_source = QComboBox()
        self.yaw_source.addItem("열림 방향 정렬 (그리퍼 X = 여는 쪽)", "opening")
        self.yaw_source.addItem("OBB 장축 정렬", "obb_long")
        self.yaw_source.addItem("고정", "fixed")
        self.yaw_source.setToolTip("vertical 접근일 때 수직축 회전 A 를 무엇에 맞출지.")
        yaw_form.addRow("회전 기준", self.yaw_source)

        self.a_fixed = QDoubleSpinBox()
        self.a_fixed.setRange(-180, 180)
        self.a_fixed.setDecimals(2)
        self.a_fixed.setSuffix(" °")
        yaw_form.addRow("고정 A (회전=고정)", self.a_fixed)

        self.a_offset = QDoubleSpinBox()
        self.a_offset.setRange(-180, 180)
        self.a_offset.setDecimals(2)
        self.a_offset.setSuffix(" °")
        self.a_offset.setToolTip(
            "이미지에서 잰 방향 각도를 base yaw(A)로 바꿀 때 더하는 보정 상수.\n카메라 장착 방향에 따른 고정 오프셋 — 한 번 맞춰두면 됨."
        )
        yaw_form.addRow("A 보정 오프셋", self.a_offset)

        # --- Z축 회전 제한 (호스/케이블 감김 방지) ---
        self.yaw_limit_chk = QCheckBox("Z 회전 제한 사용")
        self.yaw_limit_chk.setToolTip(
            "진공 호스가 팔을 따라 붙어 있어 손목이 계속 한 방향으로 돌면 줄이 감긴다.\n"
            "A(Z축 회전)를 아래 범위 안으로 강제하고, 못 맞추는 파지는 자동 배제한다.\n"
            "끄면 예전처럼 제한 없이 계산한다 (감김 위험)."
        )
        yaw_form.addRow(self.yaw_limit_chk)

        self.yaw_min = QDoubleSpinBox()
        self.yaw_min.setRange(-720, 720)
        self.yaw_min.setDecimals(1)
        self.yaw_min.setSuffix(" °")
        self.yaw_max = QDoubleSpinBox()
        self.yaw_max.setRange(-720, 720)
        self.yaw_max.setDecimals(1)
        self.yaw_max.setSuffix(" °")
        lim_row = QHBoxLayout()
        lim_row.addWidget(self.yaw_min)
        lim_row.addWidget(QLabel("~"))
        lim_row.addWidget(self.yaw_max)
        lim_row.addStretch()
        lim_w = QWidget()
        lim_row.setContentsMargins(0, 0, 0, 0)
        lim_w.setLayout(lim_row)
        yaw_form.addRow("허용 A 범위", lim_w)

        self.yaw_allow_180 = QCheckBox("180° 뒤집힌 자세도 허용")
        self.yaw_allow_180.setToolTip(
            "흡착 패드는 원형이라 툴을 Z축으로 180° 뒤집어도 같은 지점을 잡는다.\n"
            "허용하면 범위를 맞출 선택지가 넓어져 배제되는 파지가 줄어든다.\n"
            "다만 툴 X 방향이 반대가 되므로 '여는 방향 정렬'이 중요하면 끈다."
        )
        yaw_form.addRow(self.yaw_allow_180)
        root.addWidget(yaw_group)

        # --- 티칭 ---
        teach_row = QHBoxLayout()
        self.btn_teach = QPushButton("📍 현재 로봇 자세로 Z·B·C 고정")
        self.btn_teach.setToolTip("로봇을 원하는 top-down 파지 자세로 jog 한 뒤 누르면,\n현재 TCP 의 Z→작업평면 Z·파지 Z, B·C→고정 B·C 로 채운다.")
        self.btn_teach.clicked.connect(self._teach_from_current)
        teach_row.addWidget(self.btn_teach)
        teach_row.addStretch()
        root.addLayout(teach_row)

        # --- OK/Cancel ---
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._load(self.cfg)
        # 방식에 따라 관련 없는 필드 흐리게
        self.pos_mode.currentIndexChanged.connect(self._update_enabled)
        self.approach.currentIndexChanged.connect(self._update_enabled)
        self.yaw_source.currentIndexChanged.connect(self._update_enabled)
        self.yaw_limit_chk.stateChanged.connect(self._update_enabled)
        self._update_enabled()

    @staticmethod
    def _set_combo(combo: QComboBox, data):
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _load(self, cfg: Dict):
        self._set_combo(self.pos_mode, cfg.get("pos_mode", "cloud"))
        self._set_combo(self.xy_source, cfg.get("xy_source", "obb_center"))
        self.z_plane.setValue(cfg.get("z_plane", 250.0))
        self.z_pick.setValue(cfg.get("z_pick", 250.0))
        self._set_combo(self.approach, cfg.get("approach", "normal"))
        self.b_fixed.setValue(cfg.get("b_fixed", 0.0))
        self.c_fixed.setValue(cfg.get("c_fixed", 180.0))
        self._set_combo(self.yaw_source, cfg.get("yaw_source", "opening"))
        self.a_fixed.setValue(cfg.get("a_fixed", 0.0))
        self.a_offset.setValue(cfg.get("a_offset", 0.0))
        self.yaw_limit_chk.setChecked(bool(cfg.get("yaw_limit", True)))
        self.yaw_min.setValue(cfg.get("yaw_min", -180.0))
        self.yaw_max.setValue(cfg.get("yaw_max", 180.0))
        self.yaw_allow_180.setChecked(bool(cfg.get("yaw_allow_180", False)))

    def _update_enabled(self):
        is_plane = self.pos_mode.currentData() == "plane"
        is_vert = self.approach.currentData() == "vertical"
        self.xy_source.setEnabled(is_plane)
        self.z_plane.setEnabled(is_plane or is_vert)  # 회전 방향 투영에도 평면 필요
        self.z_pick.setEnabled(is_plane)
        self.b_fixed.setEnabled(is_vert)
        self.c_fixed.setEnabled(is_vert)
        self.yaw_source.setEnabled(is_vert)
        self.a_fixed.setEnabled(is_vert and self.yaw_source.currentData() == "fixed")
        self.a_offset.setEnabled(is_vert and self.yaw_source.currentData() != "fixed")
        on = self.yaw_limit_chk.isChecked()
        self.yaw_min.setEnabled(on)
        self.yaw_max.setEnabled(on)
        self.yaw_allow_180.setEnabled(on)

    def _teach_from_current(self):
        if self._get_tcp is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되어 있지 않습니다")
            return
        tcp = self._get_tcp()
        if not tcp:
            QMessageBox.warning(self, "오류", "현재 로봇 자세를 읽지 못했습니다")
            return
        self.z_plane.setValue(float(tcp["z"]))
        self.z_pick.setValue(float(tcp["z"]))
        self.b_fixed.setValue(float(tcp["b"]))
        self.c_fixed.setValue(float(tcp["c"]))
        QMessageBox.information(
            self,
            "티칭 완료",
            f"현재 TCP 로 채웠습니다.\nZ={tcp['z']:.1f}mm  B={tcp['b']:.2f}°  C={tcp['c']:.2f}°",
        )

    def result_config(self) -> Dict:
        return {
            "pos_mode": self.pos_mode.currentData(),
            "xy_source": self.xy_source.currentData(),
            "z_plane": self.z_plane.value(),
            "z_pick": self.z_pick.value(),
            "approach": self.approach.currentData(),
            "b_fixed": self.b_fixed.value(),
            "c_fixed": self.c_fixed.value(),
            "yaw_source": self.yaw_source.currentData(),
            "a_fixed": self.a_fixed.value(),
            "a_offset": self.a_offset.value(),
            "yaw_limit": self.yaw_limit_chk.isChecked(),
            "yaw_min": self.yaw_min.value(),
            "yaw_max": self.yaw_max.value(),
            "yaw_allow_180": self.yaw_allow_180.isChecked(),
        }


# ============================================================
# Bin Picking 탭
# ============================================================


class BinPickingTab(VisionTabMixin, RobotControlMixin, QWidget):
    """
    Bin Picking 통합 탭
    - 캡처 → 객체 탐지 → 포즈 계산 → 로봇 좌표 변환
    """

    # RF-DETR TensorRT 엔진 경로. 환경변수 RFDETR_MODEL_PATH 로 override 가능.
    # det / seg 엔진 모두 지원 — 결과에 mask 가 있으면(seg) 자동으로 mask 픽셀의 XYZ 로
    # 중심을 계산하고 2D 뷰에 mask 를 칠한다. mask 가 없으면(det) bbox 사각형 크롭으로 폴백.
    # 아래 두 경로 중 원하는 쪽을 활성화 (또는 RFDETR_MODEL_PATH 환경변수로 지정).
    RFDETR_MODEL_PATH = os.environ.get(
        "RFDETR_MODEL_PATH",
        # "/home/robotegra/michael/rf-detr/tmp/rfdetr-det-nano/trt/rfdetr-nano.engine",       # det: bbox 만
        "/home/robotegra/michael/rf-detr/tmp/rfdetr-seg-nano/trt/rfdetr-seg-nano.engine",  # seg: mask 사용
    )

    # 모델 클래스 id → 이름 매핑 (rf-detr/tmp/inference.py 의 CLASSES 와 동일).
    RFDETR_CLASSES = {
        0: "ladybug",
        1: "heart",
        2: "wings",
    }

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
        self.detections = []  # RF-DETR 탐지 리스트
        self.pick_objects = []  # 포즈 계산된 객체
        self.selected_idx = None
        self.target_pose = None  # 선택된 객체의 로봇 base 좌표계 자세

        # T_calib/calib_mode 는 VisionTabMixin 프로퍼티 (main.calib 소유)

        # 연속 픽(자동 반복) 상태
        self._auto_running = False
        self._auto_done = 0

        # Bin Box (작업 볼륨) — 충돌 방지용. **로봇 base 좌표계**에 정의한다.
        #   빈의 벽은 중력(base Z)에 평행하고 그리퍼 자세도 base 이므로 base 가 자연스럽다.
        #   (카메라 AABB 는 카메라가 기울면 실제 상자와 어긋남 → docs/bin_picking.md 참고)
        #   3D 뷰는 카메라 좌표계라, 표시할 땐 8 코너를 base→cam 으로 역변환해 그린다.
        # None = 미설정. dict 구조는 _default_bin_box() 참고.
        self.bin_box = None
        self._bin_box_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin_box.json")
        self._load_bin_box()
        self._grasp_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_config.json")

        # Grasp(파지) 설정 — 검출된 객체를 어떻게 잡을지 규칙. 기본값은 기존 동작
        # (3D 중심 + 표면 법선). 투명/평면 객체는 다이얼로그에서 아래로 바꾼다:
        #   위치 plane(고정 평면에 광선 투영) + 접근 vertical(수직) + 회전 opening(열림 방향).
        # 자세한 의미는 _compute_grasp_tcp / GraspConfigDialog 참고.
        self.grasp_config = {
            "pos_mode": "cloud",  # cloud=3D 중심(기존) | plane=고정 평면 광선 투영(깊이 불사용)
            "xy_source": "obb_center",  # plane 모드에서 투영할 픽셀: obb_center | mask_center | bbox_center
            "z_plane": 250.0,  # 광선 투영용 base 평면 Z(mm) — 보통 케이스 윗면 높이
            "z_pick": 250.0,  # 실제 파지 Z(mm)
            "approach": "normal",  # normal=표면 법선(기존) | vertical=수직 top-down
            "b_fixed": 0.0,  # vertical 일 때 고정 B(deg)
            "c_fixed": 180.0,  # vertical 일 때 고정 C(deg)
            "yaw_source": "opening",  # vertical 일 때 A 결정: opening | obb_long | fixed
            "a_fixed": 0.0,  # yaw_source=fixed 일 때 A(deg)
            "a_offset": 0.0,  # 이미지 각도 → base yaw 보정 상수(deg)
            # Z축 회전(A) 제한 — 진공 호스/케이블이 손목에 감기는 것을 막는다.
            # ABC unwrap 은 "현재 자세에 가장 가까운 표현"을 고르므로 A 가 190°, 200°…
            # 로 누적될 수 있다. 아래 범위를 벗어나면 같은 자세의 다른 표현(±360,
            # 선택 시 ±180)으로 되돌리고, 그래도 안 되면 그 파지를 배제한다.
            # 기본은 **끔**. 켜면 범위 밖 각도의 객체는 파지 후보에서 배제된다.
            # 주의: 이 제한만으로는 호스 감김을 못 막는다. 손목이 어느 쪽으로 도는지는
            # 우리가 보내는 Cartesian A 가 아니라 로봇의 관절 해(解) 선택이 정하기 때문이다
            # (A=-170° 와 +190° 는 같은 자세). 감김 제어가 필요하면 $AXIS_ACT 로 A6 를
            # 직접 읽어 관절 제어(add_move_axis)로 가야 한다 — 미구현.
            "yaw_limit": False,
            "yaw_min": -180.0,  # 허용 A 최소(deg) — 제한을 켰을 때만 의미
            "yaw_max": 180.0,  # 허용 A 최대(deg)
            # 180° 뒤집힌 자세 허용 여부. 흡착 패드는 원형이라 파지 지점은 같지만
            # **툴 X 방향이 반대**가 되어 '여는 방향 정렬'이 뒤집힌다. 배치 방향
            # 일관성이 이 시스템의 목적이므로 기본은 끔 — 범위를 못 맞추는 파지는
            # 뒤집는 대신 그냥 건너뛴다. 방향이 상관없으면 켜서 성공률을 올릴 수 있다.
            "yaw_allow_180": False,
        }
        self._load_grasp_config()  # 저장된 설정 복원 (Z 회전 제한 등 안전 설정이 재시작에 살아남도록)

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

        self.btn_detect_obb = QPushButton("OBB 검출")
        self.btn_detect_obb.setToolTip(
            "검출된 각 객체의 마스크에 cv2.minAreaRect 를 적용해\n"
            "회전 사각형(OBB: 중심·크기·회전각)을 구한다. 마스크가 있는\n"
            "검출(객체 검출/SAM3 검출)에만 동작."
        )
        self.btn_detect_obb.clicked.connect(self._detect_obb)
        self.btn_detect_obb.setStyleSheet("background-color: #00838F; color: white; font-weight: bold;")
        top_row.addWidget(self.btn_detect_obb)

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

        # === 2행: SAM3 텍스트 프롬프트 검출 (검출기 없이 텍스트만으로 분할) ===
        sam3_row = QHBoxLayout()
        sam3_row.addWidget(QLabel("SAM3 텍스트:"))
        self.sam3_prompt_input = QLineEdit("rectangular")
        self.sam3_prompt_input.setPlaceholderText("검출할 객체를 영어 명사구로 (예: cosmetic case, transparent box)")
        self.sam3_prompt_input.setToolTip(
            "SAM 3 개념 분할 — 이 텍스트에 해당하는 모든 객체를 캡처 이미지에서 찾아\n"
            "분할 마스크를 만든다. Grounding DINO 없이 SAM 3 하나로 동작.\n"
            "Conf 값이 점수 임계값으로 함께 적용됨."
        )
        self.sam3_prompt_input.returnPressed.connect(self._detect_sam3)
        self.sam3_prompt_input.setFixedWidth(400)  # stretch 대신 고정 너비
        sam3_row.addWidget(self.sam3_prompt_input)

        self.btn_detect_sam3 = QPushButton("SAM3 검출")
        self.btn_detect_sam3.clicked.connect(self._detect_sam3)
        self.btn_detect_sam3.setStyleSheet("background-color: #6A1B9A; color: white; font-weight: bold;")
        sam3_row.addWidget(self.btn_detect_sam3)

        sam3_row.addStretch()  # 남는 공간은 오른쪽으로 (입력창 늘어나지 않게)

        sam3_widget = QWidget()
        sam3_widget.setLayout(sam3_row)
        sam3_widget.setFixedHeight(sam3_widget.sizeHint().height())
        layout.addWidget(sam3_widget)

        # === 3행: 여는 방향(클램셸 힌지/뚜껑) 추정 — 방식 선택 + 조정값 ===
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("여는 방향 방식:"))
        self.opening_method_combo = QComboBox()
        # userData = 내부 식별자. 내부 격자 비대칭이 가장 강건해서 맨 위 = 기본값.
        self.opening_method_combo.addItem("내부 격자 비대칭", "grid")
        self.opening_method_combo.addItem("이음선 에지", "seam")
        self.opening_method_combo.addItem("내부 밝기 비대칭", "brightness")
        self.opening_method_combo.setToolTip(
            "내부 격자 비대칭: 투명 케이스 내부 칸 배열이 한쪽으로 치우친 걸 이용 — OBB로\n"
            "  똑바로 세워 격자 밴드를 찾고 위/아래 여백 크기로 방향 (가장 강건, 권장).\n"
            "이음선 에지: 뚜껑-바닥 이음선(여는 쪽+양옆 U자)의 에지 무게중심으로 판별.\n"
            "내부 밝기 비대칭: 투명 케이스 내부(팬/거울/힌지)의 밝기 무게중심으로 판별.\n"
            "모든 방식 여는 축은 OBB 단축으로 고정, 부호(어느 긴 변이 립인지)만 정함."
        )
        self.opening_method_combo.currentIndexChanged.connect(self._update_opening_settings_visibility)
        op_row.addWidget(self.opening_method_combo)

        # 침식% (이음선/밝기 방식 전용) — 라벨+스핀을 한 위젯으로 묶어 통째로 show/hide
        op_row.addSpacing(10)
        self.opening_erode_spin = QSpinBox()
        self.opening_erode_spin.setRange(0, 25)
        self.opening_erode_spin.setValue(0)
        self.opening_erode_spin.setFixedWidth(60)
        self.opening_erode_spin.setToolTip(
            "마스크를 단축의 이 %만큼 침식해 외곽 실루엣 에지를 제외한다.\n"
            "실루엣이 새어들면 키우고, 내부 이음선까지 깎이면 줄인다. (이음선/밝기 방식)"
        )
        self.opening_erode_widget = self._labeled_widget("침식%:", self.opening_erode_spin)
        op_row.addWidget(self.opening_erode_widget)

        # 에지 임계% (이음선 방식 전용)
        op_row.addSpacing(10)
        self.opening_thr_spin = QSpinBox()
        self.opening_thr_spin.setRange(0, 95)
        self.opening_thr_spin.setValue(0)
        self.opening_thr_spin.setFixedWidth(60)
        self.opening_thr_spin.setToolTip(
            "이음선 방식 전용: 지지영역 에지 크기의 이 백분위 미만은 0으로 버려\n"
            "약한 텍스처 노이즈를 억제한다. 0 = 사용 안 함. (예: 70 → 상위 30% 에지만)"
        )
        self.opening_thr_widget = self._labeled_widget("에지 임계%:", self.opening_thr_spin)
        op_row.addWidget(self.opening_thr_widget)

        # 격자 임계% (격자 방식 전용) — 세로벽 프로파일에서 격자 밴드로 인정할 기준
        op_row.addSpacing(10)
        self.opening_grid_thr_spin = QSpinBox()
        self.opening_grid_thr_spin.setRange(10, 90)
        self.opening_grid_thr_spin.setValue(45)
        self.opening_grid_thr_spin.setFixedWidth(60)
        self.opening_grid_thr_spin.setToolTip(
            "격자 방식 전용: 세로벽 밀도 프로파일 최댓값의 이 % 이상인 구간을 '격자 밴드'로\n"
            "잡는다. 밴드가 빈 여백까지 삼키면 올리고, 격자 일부만 잡으면 내린다. (기본 40)"
        )
        self.opening_grid_thr_widget = self._labeled_widget("격자 임계%:", self.opening_grid_thr_spin)
        op_row.addWidget(self.opening_grid_thr_widget)

        # 옆벽 크롭% (격자 방식 전용) — 케이스 투명 옆벽(세로 에지 오염) 좌우 제거
        op_row.addSpacing(10)
        self.opening_grid_crop_spin = QSpinBox()
        self.opening_grid_crop_spin.setRange(0, 40)
        self.opening_grid_crop_spin.setValue(15)
        self.opening_grid_crop_spin.setFixedWidth(60)
        self.opening_grid_crop_spin.setToolTip(
            "격자 방식 전용: 케이스 좌우(투명 옆벽)를 이 %만큼 잘라낸다. 옆벽 세로 에지가\n"
            "모든 행을 오염시키므로 제거. 프레임이 두꺼우면 키운다. (기본 15, 상하는 4% 고정)"
        )
        self.opening_grid_crop_widget = self._labeled_widget("옆벽 크롭%:", self.opening_grid_crop_spin)
        op_row.addWidget(self.opening_grid_crop_widget)

        op_row.addSpacing(10)
        self.opening_invert_chk = QCheckBox("방향 반전")  # 모든 방식 공통
        self.opening_invert_chk.setToolTip("추정된 여는 방향 벡터를 180° 뒤집는다 (부호 규칙이 제품과 반대일 때).")
        op_row.addWidget(self.opening_invert_chk)

        self.btn_detect_opening = QPushButton("여는 방향")
        self.btn_detect_opening.setToolTip("위 방식·조정값으로 각 객체의 여는 쪽을 추정해 화살표로 표시. 마스크 필요.")
        self.btn_detect_opening.clicked.connect(self._detect_opening)
        self.btn_detect_opening.setStyleSheet("background-color: #EF6C00; color: white; font-weight: bold;")
        op_row.addWidget(self.btn_detect_opening)

        op_row.addStretch()
        self._update_opening_settings_visibility()  # 기본(격자) 방식에 맞춰 초기 표시

        self.btn_grasp_config = QPushButton("⚙ Grasp 설정")
        self.btn_grasp_config.setToolTip(
            "검출된 객체를 어떻게 잡을지(파지점·접근 방향·회전) 규칙을 설정한다.\n"
            "투명/평면 객체는 여기서 '고정 평면 + 수직 접근 + 열림 방향 정렬'로 바꾼다."
        )
        self.btn_grasp_config.clicked.connect(self._open_grasp_config)
        self.btn_grasp_config.setStyleSheet("background-color: #455A64; color: white; font-weight: bold;")
        op_row.addWidget(self.btn_grasp_config)

        self.btn_bin_box = QPushButton("📦 Bin Box")
        self.btn_bin_box.setToolTip(
            "빈(bin) 상자를 로봇 base 좌표계로 정의 — 충돌 방지의 기준.\n"
            "ROI 에서 자동 산출 / 수치 입력 / 로봇 티칭(높이)으로 설정.\n"
            "3D 뷰에 노란 상자(외곽)와 초록 상자(파지 허용 영역)로 표시된다."
        )
        self.btn_bin_box.clicked.connect(self._open_bin_box_config)
        self.btn_bin_box.setStyleSheet("background-color: #5D4037; color: white; font-weight: bold;")
        op_row.addWidget(self.btn_bin_box)

        op_widget = QWidget()
        op_widget.setLayout(op_row)
        op_widget.setFixedHeight(op_widget.sizeHint().height())
        layout.addWidget(op_widget)

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
        # 축 표시가 세로→가로(2×3)로 줄어든 만큼 검출 테이블을 세로로 늘림
        info_layout.addWidget(det_group, stretch=1)

        # 선택된 객체 정보 — X Y Z / A B C 를 2행×3열 그리드로 가로 배치 (공간 축소)
        sel_group = QGroupBox("선택된 피킹 포즈 (로봇 base 좌표)")
        sel_layout = QGridLayout(sel_group)
        sel_layout.setHorizontalSpacing(12)
        self.robot_labels = {}
        for i, axis in enumerate(["X", "Y", "Z", "A", "B", "C"]):
            r, c = divmod(i, 3)  # 0~2 → 0행, 3~5 → 1행
            cell = QHBoxLayout()
            cell.addWidget(QLabel(f"{axis}:"))
            lab = QLabel("---")
            lab.setStyleSheet("font-family: monospace; font-size: 15px; font-weight: bold; color: #0066cc;")
            self.robot_labels[axis] = lab
            cell.addWidget(lab)
            cell.addStretch()
            sel_layout.addLayout(cell, r, c)
        info_layout.addWidget(sel_group)

        # 로봇 이동 제어 + 시퀀스 큐 — 레이아웃은 RobotControlMixin 이 소유한다.
        # (탭마다 복붙하지 않으므로 여기서 버튼을 옮기면 다른 탭에도 같이 반영됨)
        info_layout.addWidget(self._build_move_group())
        info_layout.addWidget(self._build_seq_group())

        # === 연속 픽 (자동 반복) — 캡처→검출→선택→픽→놓기 를 빌 때까지 반복 ===
        auto_group = QGroupBox("연속 픽 (자동 반복)")
        auto_layout = QVBoxLayout(auto_group)

        crit_row = QHBoxLayout()
        crit_row.addWidget(QLabel("선택 기준:"))
        self.auto_crit_combo = QComboBox()
        self.auto_crit_combo.addItem("가장 위 (Z 최대)", "topmost")
        self.auto_crit_combo.addItem("검출 신뢰도 최고", "conf")
        self.auto_crit_combo.addItem("여는 방향 신뢰도 최고", "opening")
        self.auto_crit_combo.addItem("빈 중심에 가까움", "center")
        self.auto_crit_combo.setToolTip(
            "매 사이클에서 어떤 객체를 집을지 고르는 기준.\n"
            "가장 위: 빈피킹 표준 — 아래 것부터 건드리면 더미가 무너진다.\n"
            "빈 중심: 벽에서 먼 것부터 (Bin Box 설정 시 그 중심 기준)."
        )
        crit_row.addWidget(self.auto_crit_combo, stretch=1)
        auto_layout.addLayout(crit_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("최대 반복:"))
        self.auto_max_spin = QSpinBox()
        self.auto_max_spin.setRange(1, 200)
        self.auto_max_spin.setValue(10)
        self.auto_max_spin.setToolTip("폭주 방지 상한. 이 횟수만큼 픽하면 자동 종료된다.")
        max_row.addWidget(self.auto_max_spin)
        max_row.addStretch()
        self.auto_progress_label = QLabel("대기 중")
        self.auto_progress_label.setStyleSheet("color: #555;")
        max_row.addWidget(self.auto_progress_label)
        auto_layout.addLayout(max_row)

        btn_row = QHBoxLayout()
        self.btn_auto_start = QPushButton("▶ 연속 픽 시작")
        self.btn_auto_start.setMinimumHeight(40)
        self.btn_auto_start.setStyleSheet("font-weight: bold; background-color: #00695C; color: white;")
        self.btn_auto_start.setToolTip(
            "캡처 → SAM3 검출 → 여는 방향 → 자동 선택 → 픽 → 놓기 → Home 을\n"
            "검출이 없을 때까지(또는 최대 반복까지) 자동 반복한다.\n"
            "⚠ 로봇이 자율로 움직인다 — Space 또는 정지 버튼으로 즉시 중단 가능."
        )
        self.btn_auto_start.clicked.connect(self._auto_start)
        self.btn_auto_start.setEnabled(False)
        btn_row.addWidget(self.btn_auto_start)

        self.btn_auto_stop = QPushButton("⏹ 정지")
        self.btn_auto_stop.setMinimumHeight(40)
        self.btn_auto_stop.setStyleSheet("font-weight: bold; background-color: #C62828; color: white;")
        self.btn_auto_stop.setToolTip("현재 사이클을 마치는 대로 반복을 멈춘다 (즉시 정지는 Space 비상정지)")
        self.btn_auto_stop.clicked.connect(lambda: self._auto_stop("사용자 정지"))
        self.btn_auto_stop.setEnabled(False)
        btn_row.addWidget(self.btn_auto_stop)
        auto_layout.addLayout(btn_row)

        info_layout.addWidget(auto_group)

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

        # 모드 표시 자동 갱신 (2초마다)
        self._mode_timer = QTimer(self)
        self._mode_timer.timeout.connect(self._refresh_mode_display)
        self._mode_timer.start(2000)

    @staticmethod
    def _labeled_widget(label_text: str, widget) -> QWidget:
        """라벨 + 위젯을 한 컨테이너로 묶어 통째로 show/hide 할 수 있게 한다."""
        cont = QWidget()
        lay = QHBoxLayout(cont)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label_text))
        lay.addWidget(widget)
        return cont

    def _update_opening_settings_visibility(self):
        """여는 방향 방식에 따라 관련 있는 조정값만 표시.
        침식%=이음선/밝기, 에지 임계%=이음선, 격자 임계%·옆벽 크롭%=격자 (반전은 공통)."""
        method = self.opening_method_combo.currentData()
        self.opening_erode_widget.setVisible(method in ("seam", "brightness"))
        self.opening_thr_widget.setVisible(method == "seam")
        self.opening_grid_thr_widget.setVisible(method == "grid")
        self.opening_grid_crop_widget.setVisible(method == "grid")

    def _switch_view(self, idx: int):
        self.view_stack.setCurrentIndex(idx)
        self.btn_view_2d.setChecked(idx == 0)
        self.btn_view_3d.setChecked(idx == 1)
        # 3D로 전환 시 위젯 크기가 확정된 후 카메라 재적용
        if idx == 1:
            # Qt가 리사이즈를 처리한 뒤 카메라 적용 (지연)
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self.view_3d.refresh_camera)

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
        self.btn_move.setEnabled(False)
        self.det_table.setRowCount(0)
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.robot_labels[axis].setText("---")

        # 2D 뷰 갱신
        self.view_2d.set_image(image)
        self.view_2d.set_boxes([])
        self.view_2d.set_masks([])
        self.view_2d.set_obbs([])
        self.view_2d.set_arrows([])
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
        self._refresh_bin_box_views()  # 3D 는 clear 됐고 2D 도 새 이미지라 Bin Box 재표시
        self.view_3d.reset_view()

        self.main.statusBar().showMessage("캡처 완료")

    def _on_roi_dragged(self, x1: int, y1: int, x2: int, y2: int):
        """2D 드래그 = **Bin Box(작업 볼륨) 설정**.

        실제 빈피킹 프로그램처럼 ROI 와 빈 상자를 하나로 본다. 드래그한 영역의 3D 점을
        base 로 옮겨 XY·yaw 를 산출하고, 그 결과를 곧바로 Bin Box 로 저장·표시한다.
        캘리브레이션이 없으면 base 변환을 못 하므로 기존 ROI 상자 동작으로 폴백.
        """
        self.roi_2d = (x1, y1, x2, y2)

        # 2D bbox 안의 3D 포인트들의 범위 → 3D ROI (깊이 밴드 2차 필터용, 유지)
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

        # 드래그 결과로 Bin Box 갱신 (ROI == Bin Box)
        bb = self._bin_box_from_drag()
        if bb is not None:
            self.bin_box = bb
            self._save_bin_box()
            self._refresh_bin_box_views()
            self.main.statusBar().showMessage(
                f"Bin Box 설정: 중심({bb['cx']:.0f},{bb['cy']:.0f}) 크기({bb['sx']:.0f}×{bb['sy']:.0f}) "
                f"yaw {bb['yaw']:.1f}° Z[{bb['z_floor']:.0f}~{bb['z_rim']:.0f}]mm  — 높이는 📦 Bin Box 에서 티칭 권장"
            )
        elif self.roi_3d is not None:
            self.main.statusBar().showMessage(
                f"ROI 설정 (캘리브레이션 없어 Bin Box 미생성): " f"Z[{self.roi_3d['z_min']:.0f},{self.roi_3d['z_max']:.0f}] mm"
            )

    def _apply_roi_to_3d(self):
        if self.roi_3d is None:
            return
        # Bin Box 가 설정돼 있으면 그게 작업 볼륨의 기준이다. ROI 자동 상자(카메라 AABB)와
        # 겹쳐 노란 박스가 두 개로 보이므로 ROI 쪽은 그리지 않는다.
        # (ROI 영역 자체는 2D 뷰의 노란 사각형으로 계속 확인 가능)
        if self.bin_box:
            self.view_3d.remove_named("roi_box")
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
        """ROI 해제 = **Bin Box 완전 삭제** (ROI 와 Bin Box 는 같은 것이므로).

        2D 표시·3D 상자·메모리 값·저장 파일까지 모두 지운다.
        """
        self.roi_2d = None
        self.roi_3d = None
        self.bin_box = None
        self.view_2d.set_roi(None)
        self.view_2d.set_roi_polygon(None)
        for name in ("roi_box", "bin_box", "bin_safe"):
            self.view_3d.remove_named(name)
        try:
            self.view_3d.plotter.render()
        except Exception:
            pass
        # 저장 파일도 삭제 → 재시작해도 안 살아남음
        try:
            if os.path.exists(self._bin_box_path):
                os.remove(self._bin_box_path)
        except Exception as e:
            logger.warning(f"Bin Box 파일 삭제 실패: {e}")
        self.main.statusBar().showMessage("ROI/Bin Box 해제 — 저장값도 삭제됨")

    def _detect(self):
        """RF-DETR 객체 검출. 실제 추론은 object_detector.RFDetrDetector 가 담당하고,
        여기선 위젯 값 읽기 + 상태 표시 + 에러 다이얼로그만 처리한다."""
        if self.current_image is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return

        self.main.statusBar().showMessage("객체 탐지 중...")
        QApplication.processEvents()

        from object_detector import RFDetrDetector, DetectorUnavailable, DetectorError

        # 검출기 캐싱 (엔진 로드는 비싸므로 1회만)
        if not hasattr(self, "_rfdetr"):
            self._rfdetr = RFDetrDetector(self.RFDETR_MODEL_PATH, self.RFDETR_CLASSES, RFDETR_DETECTOR_DIR)
        try:
            detections, infer_ms = self._rfdetr.detect(self.current_image, self.conf_spin.value())
        except (DetectorUnavailable, DetectorError) as e:
            QMessageBox.critical(self, "오류", str(e))
            return

        self._apply_detections(detections, "검출", infer_ms=infer_ms)

    def _detect_sam3(self):
        """SAM 3 텍스트 프롬프트 검출. 추론은 object_detector.Sam3Detector 가 담당.

        Grounding DINO 없이 명사구 하나로 개념 분할. 결과는 RF-DETR 경로와
        동일한 detections 포맷이라 공통 _apply_detections 로 넘어간다.
        """
        if self.current_image is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return
        prompt = self.sam3_prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "오류", "검출할 객체를 설명하는 텍스트를 입력하세요 (예: cosmetic case)")
            return

        from object_detector import Sam3Detector, DetectorUnavailable, DetectorError

        if not hasattr(self, "_sam3"):
            self._sam3 = Sam3Detector(SAM3_MODEL_DIR or None)

        # 최초 1회 모델 로드는 수십 초 걸리므로 로드 안내 먼저 표시
        if not self._sam3.loaded:
            self.main.statusBar().showMessage("SAM3 모델 로드 중... (최초 1회, 수십 초 소요 가능)")
        else:
            self.main.statusBar().showMessage(f"SAM3 검출 중... ('{prompt}')")
        QApplication.processEvents()

        conf = self.conf_spin.value()
        try:
            detections, infer_ms = self._sam3.detect(self.current_image, conf, prompt=prompt)
        except (DetectorUnavailable, DetectorError) as e:
            QMessageBox.critical(self, "오류", str(e))
            return

        if not detections:
            self.main.statusBar().showMessage(
                f"SAM3: '{prompt}' 에 해당하는 객체를 못 찾음 (Conf {conf:.2f} 이상, 추론 {infer_ms:.0f}ms). 텍스트/Conf 조정"
            )
        self._apply_detections(detections, f"SAM3 검출 ('{prompt}')", infer_ms=infer_ms)

    # 검출된 각 객체의 팔레트 색 (bbox/mask 표시와 동일 규칙: 유효 픽 객체를 draw 순으로)
    _OBJ_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80), (255, 80, 255), (80, 255, 255)]

    def _object_color_map(self) -> Dict[int, tuple]:
        """검출 index → 표시 색. bbox/mask/OBB 오버레이가 같은 색을 쓰도록 한 곳에서 계산."""
        valid_indices = {o["index"] for o in self.pick_objects}
        cmap = {}
        draw_i = 0
        for i in range(len(self.detections)):
            if i not in valid_indices:
                continue
            cmap[i] = self._OBJ_COLORS[draw_i % len(self._OBJ_COLORS)]
            draw_i += 1
        return cmap

    def _detect_obb(self):
        """검출된 각 객체의 마스크에 cv2.minAreaRect 를 적용해 OBB(회전 사각형)를 구한다.

        검출 방식(객체 검출/SAM3)과 무관하게 det["mask"](H×W bool)가 있으면 동작.
        결과는 det["obb"] = {center(px), size(px), angle(deg), box_pts(4×2)} 로 저장하고
        2D 뷰에 회전 사각형을 그린다. bbox 만 있고 마스크가 없는 검출은 건너뛴다.
        """
        if not self.detections:
            QMessageBox.warning(self, "오류", "먼저 객체를 검출하세요 (객체 검출 또는 SAM3 검출)")
            return

        cmap = self._object_color_map()  # bbox 와 동일 색
        obb_overlays = []
        n_ok = 0
        n_no_mask = 0
        for i, det in enumerate(self.detections):
            mask = det.get("mask")
            if mask is None:
                det.pop("obb", None)
                n_no_mask += 1
                continue
            obb = oa.obb_from_mask(mask)
            if obb is None:
                det.pop("obb", None)
                continue
            det["obb"] = obb
            color = cmap.get(i, (200, 200, 200))  # 유효 픽 객체가 아니면 회색
            obb_overlays.append((np.asarray(obb["box_pts"]), color, i, obb["angle"]))
            n_ok += 1

        self.view_2d.set_obbs(obb_overlays)
        if n_ok == 0:
            msg = "OBB 를 구할 마스크가 없습니다 (마스크 없는 검출뿐)" if n_no_mask else "OBB 검출 실패 (윤곽선 없음)"
            self.main.statusBar().showMessage(msg)
            QMessageBox.warning(self, "OBB 검출", msg + "\n\nRF-DETR seg 모델 또는 SAM3 검출처럼 마스크가 있는 검출이 필요합니다.")
        else:
            skip = f", 마스크 없음 {n_no_mask}개 건너뜀" if n_no_mask else ""
            self.main.statusBar().showMessage(f"OBB 검출 완료: {n_ok}개{skip}")

    def _detect_opening(self):
        """검출된 각 객체의 여는 방향(힌지 반대편)을 선택한 방식으로 추정.

        방식(콤보): 이음선 에지 / 내부 밝기 비대칭. 조정값: 침식%, 에지 임계%, 방향 반전.
        마스크가 있는 검출마다: OBB(없으면 즉시 계산) + 방향 → det["opening"].
        2D 뷰에 회전 사각형과 중심→여는 쪽 화살표를 함께 그린다.
        """
        if not self.detections:
            QMessageBox.warning(self, "오류", "먼저 객체를 검출하세요 (객체 검출 또는 SAM3 검출)")
            return
        img = self.current_rgb
        if img is None:
            QMessageBox.warning(self, "오류", "캡처된 이미지가 없습니다")
            return

        method = self.opening_method_combo.currentData()
        erode_ratio = self.opening_erode_spin.value() / 100.0
        thr_pct = self.opening_thr_spin.value()
        invert = self.opening_invert_chk.isChecked()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else np.asarray(img)
        # 격자 방식은 객체별 warp 라 전역 weight 맵을 안 쓴다.
        weight = None if method == "grid" else oa.opening_weight_map(gray, method, thr_pct)

        cmap = self._object_color_map()
        obb_overlays = []
        arrows = []
        n_ok = 0
        n_no_mask = 0
        low_conf = 0
        for i, det in enumerate(self.detections):
            mask = det.get("mask")
            if mask is None:
                det.pop("opening", None)
                n_no_mask += 1
                continue
            obb = det.get("obb") or oa.obb_from_mask(mask)
            if obb is None:
                continue
            det["obb"] = obb
            if method == "grid":
                res = oa.opening_from_grid(
                    mask,
                    gray,
                    obb,
                    debug=OPENING_DEBUG,
                    band_thr=self.opening_grid_thr_spin.value() / 100.0,
                    side_crop=self.opening_grid_crop_spin.value() / 100.0,
                )
            else:
                res = oa.opening_from_weight(mask, weight, obb, erode_ratio, debug=OPENING_DEBUG)
            if res is None:
                det.pop("opening", None)
                continue
            if OPENING_DEBUG and method != "grid":  # 격자는 함수 내부에서 자체 디버그 창을 띄움
                oa.debug_show_opening(self.current_rgb, det, obb, weight, res, i)
            if invert:
                dx, dy = res["dir"]
                res["dir"] = (-dx, -dy)
                res["angle_deg"] = float(np.degrees(np.arctan2(-dy, -dx)))
            res["method"] = method
            res.pop("_debug", None)  # 디버그 임시 데이터는 저장 전 제거
            det["opening"] = res
            color = cmap.get(i, (200, 200, 200))
            obb_overlays.append((np.asarray(obb["box_pts"]), color, i, obb["angle"]))
            cx, cy = obb["center"]
            dx, dy = res["dir"]
            end = (cx + dx * res["half_len"], cy + dy * res["half_len"])
            arrows.append(((cx, cy), end, color, i, res["confidence"]))
            if res["confidence"] < 0.02:
                low_conf += 1
            n_ok += 1

        self.view_2d.set_obbs(obb_overlays)
        self.view_2d.set_arrows(arrows)
        method_name = self.opening_method_combo.currentText()
        if n_ok == 0:
            msg = "여는 방향을 구할 마스크가 없습니다" if n_no_mask else "여는 방향 추정 실패"
            self.main.statusBar().showMessage(msg)
            QMessageBox.warning(self, "여는 방향", msg + "\n\n마스크가 있는 검출(객체 검출 seg / SAM3)이 필요합니다.")
        else:
            warn = f", 신뢰도 낮음 {low_conf}개(비대칭 불충분)" if low_conf else ""
            self.main.statusBar().showMessage(f"여는 방향 추정 완료({method_name}): {n_ok}개{warn}")

    def _apply_detections(self, detections: List[Dict], source: str = "검출", infer_ms: Optional[float] = None):
        """검출 결과(공통 포맷) → ROI 필터 → 3D 포즈 → 2D/3D/테이블 갱신.

        RF-DETR(_detect)와 SAM3(_detect_sam3)가 공유하는 다운스트림.
        detections 각 항목: {bbox[xyxy], confidence, class_id, class_name, (mask HxW bool)}
        """
        # ROI 필터.
        # 1차 게이트 = 사용자가 그린 2D 사각형(roi_2d) — 깊이와 무관하게 항상 적용.
        #   투명 객체(SAM3)는 중심 깊이가 NaN 이라 3D-only 필터로는 못 거르므로
        #   반드시 2D 게이트가 필요하다.
        # 2차(보조) = 중심 픽셀 깊이가 유효하고 roi_3d 가 있으면 깊이 밴드까지 확인
        #   (불투명 객체를 깊이로 분리하는 기존 장점 유지). 깊이가 NaN 이면 이 단계는
        #   건너뛰어 검출을 버리지 않는다.
        if self.roi_2d is not None:
            rx1, ry1, rx2, ry2 = self.roi_2d
            rx_lo, rx_hi = sorted((rx1, rx2))
            ry_lo, ry_hi = sorted((ry1, ry2))
            xyz = self.current_xyz
            filtered = []
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                bx_lo, bx_hi = sorted((bx1, bx2))
                by_lo, by_hi = sorted((by1, by2))
                # 1차: bbox 전체가 2D ROI 사각형 안에 포함되는가 (부분 걸침은 제외)
                if not (rx_lo <= bx_lo and bx_hi <= rx_hi and ry_lo <= by_lo and by_hi <= ry_hi):
                    continue
                # 2차: 중심 픽셀 깊이가 유효하고 roi_3d 있으면 깊이 밴드까지 확인
                if self.roi_3d is not None and xyz is not None:
                    h, w = xyz.shape[:2]
                    ix = int((bx1 + bx2) / 2.0)
                    iy = int((by1 + by2) / 2.0)
                    if 0 <= ix < w and 0 <= iy < h:
                        pt = xyz[iy, ix]
                        if not np.any(np.isnan(pt)):
                            if not (
                                self.roi_3d["x_min"] <= pt[0] <= self.roi_3d["x_max"]
                                and self.roi_3d["y_min"] <= pt[1] <= self.roi_3d["y_max"]
                                and self.roi_3d["z_min"] <= pt[2] <= self.roi_3d["z_max"]
                            ):
                                continue  # 깊이 밴드 밖 → 제외
                filtered.append(det)
            logger.info(f"ROI 필터링(2D bbox 포함{'+3D' if self.roi_3d else ''}): {len(detections)} → {len(filtered)}")
            detections = filtered

        self.detections = detections

        # 각 검출에 대해 3D 포즈 계산 (먼저 실행해서 유효 객체만 표시)
        self._compute_pick_poses()

        # 2D 뷰에 bbox 표시 (유효한 피킹 포즈가 있는 객체만)
        colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80), (255, 80, 255), (80, 255, 255)]
        valid_indices = {o["index"] for o in self.pick_objects}
        boxes = []
        mask_overlays = []  # seg 결과일 때만 채워짐 (bbox만 있으면 오버레이 없음)
        draw_i = 0  # 유효 객체만 순차 색 할당 (raw index 기준이면 팔레트가 듬성듬성 쓰여
        #            테이블/3D 순서와 색이 어긋남)
        for i, det in enumerate(detections):
            if i not in valid_indices:
                continue
            color = colors[draw_i % len(colors)]
            draw_i += 1
            label = f"{det['class_name']} {det['confidence']:.2f}"
            boxes.append((*det["bbox"], color, label, i))
            if det.get("mask") is not None:
                mask_overlays.append((det["mask"], color, i))
        self.view_2d.set_boxes(boxes)
        self.view_2d.set_masks(mask_overlays)
        self.view_2d.set_obbs([])  # 새 검출 → 이전 OBB/방향 은 무효 (버튼으로 다시 계산)
        self.view_2d.set_arrows([])

        # 테이블 갱신
        self._update_table()

        # 3D 뷰: 포인트클라우드는 유지하고 객체 마커만 갱신
        self.view_3d.show_pick_objects(self.pick_objects)

        infer_part = f", 추론 {infer_ms:.0f}ms" if infer_ms is not None else ""
        self.main.statusBar().showMessage(f"{source} 완료: {len(detections)}개, 피킹 포즈 {len(self.pick_objects)}개{infer_part}")

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

            # seg 모델이면 mask True 픽셀의 XYZ만 사용 (배경/이웃 객체 제외 → 중심 정확도↑).
            # det 모델(mask 없음)이거나 shape 불일치면 기존처럼 bbox 사각형 크롭.
            seg_mask = det.get("mask")
            if seg_mask is not None and seg_mask.shape[:2] == (h, w):
                obj_pixels = seg_mask & ~np.any(np.isnan(self.current_xyz), axis=2)
                crop = self.current_xyz[obj_pixels]
            else:
                crop_region = self.current_xyz[iy1:iy2, ix1:ix2].reshape(-1, 3)
                crop = crop_region[~np.any(np.isnan(crop_region), axis=1)]
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

        # 참고 자세(법선 접근의 ref) / eye-in-hand / 회전변화 계산용 현재 TCP
        if self.main.robot:
            cur_tcp = self.main.robot.get_tcp_position()
        else:
            cur_tcp = {"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 180}

        det = self.detections[idx] if idx < len(self.detections) else {}
        tcp, err = self._compute_grasp_tcp(obj, det, cur_tcp)
        if tcp is None:
            self.main.statusBar().showMessage(f"파지 자세 계산 실패: {err}")
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                self.robot_labels[axis].setText("---")
            self.target_pose = None
            self.btn_move.setEnabled(False)
            return

        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.robot_labels[axis].setText(f"{tcp[axis.lower()]:.2f}")

        # 이동 버튼 활성화용으로 타겟 자세 저장
        self.target_pose = tcp

        # 이동 / 시퀀스 추가 버튼은 로봇 연결이 되어 있고 타겟이 유효할 때만 활성화
        connected = self.main.robot is not None
        self.btn_move.setEnabled(connected)
        self.btn_add_obj_to_seq.setEnabled(connected)
        self.btn_add_pick_to_seq.setEnabled(connected)

        # 3D 뷰에 Tool 자세 시각화 (Tool 좌표축 + approach 지점 + 경로선)
        self._render_tcp_visualization()

        # 회전 변화량 계산 (현재 TCP와 비교) → 사용자에게 큰 관절 회전 예상 시 경고
        rot_change = self._compute_rotation_change_deg(cur_tcp, tcp)
        rot_part = f", 회전변화 {rot_change:.0f}°" if rot_change is not None else ""
        warn = "  ⚠ 큰 회전 — PTP 권장" if rot_change is not None and rot_change > 60 else ""
        cfg = self.grasp_config
        tag = f"{'평면' if cfg['pos_mode']=='plane' else '3D'}/{'수직' if cfg['approach']=='vertical' else '법선'}"
        self.main.statusBar().showMessage(f"객체 #{idx + 1} 선택 [{tag}]: X={tcp['x']:.1f}, Y={tcp['y']:.1f}, Z={tcp['z']:.1f}{rot_part}{warn}")

    # ============================================================
    # Grasp(파지) 설정 & 계산
    # ============================================================

    # ============================================================
    # 연속 픽 (자동 반복) — 캡처→검출→선택→픽→놓기→Home 을 빌 때까지
    # ============================================================

    def _auto_start(self):
        """연속 픽 시작. 로봇이 자율로 반복 동작하므로 사전 검증 + 확인을 거친다."""
        if self._auto_running:
            return
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        if self.main.camera is None or not self.main.camera.is_capture_ready:
            QMessageBox.warning(self, "오류", "카메라가 캡처 준비되지 않았습니다")
            return
        if self.T_calib is None:
            QMessageBox.warning(self, "오류", "캘리브레이션을 먼저 로드하세요")
            return
        place = getattr(self.main, "place_pose", None)
        if place is None:
            QMessageBox.warning(
                self,
                "놓기 위치 없음",
                "놓기(Place) 위치가 저장되지 않았습니다.\n" "로봇을 놓을 자리로 조그한 뒤 '📍 놓기 위치 저장'을 먼저 누르세요.",
            )
            return
        prompt = self.sam3_prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "오류", "SAM3 텍스트(검출할 객체)를 입력하세요")
            return
        if self._cycle_is_running():
            QMessageBox.warning(self, "실행 중", "다른 사이클이 실행 중입니다")
            return

        home = self.main.home_pose
        speed = self._effective_speed(self.speed_spin.value())
        msg = [
            f"▶ 연속 픽 시작 — 최대 {self.auto_max_spin.value()}회\n",
            f"검출: SAM3 '{prompt}'",
            f"선택 기준: {self.auto_crit_combo.currentText()}",
            f"속도: {speed}%" + (" (AUT 50% 상한)" if self._is_aut_mode() else ""),
            f"Approach: {self.approach_dist.value():.0f}mm, 흡착대기: {self.vac_dwell_spin.value():.1f}s",
            "",
            "각 사이클: 캡처 → SAM3 검출 → 여는 방향 → 자동 선택 →",
            "  Approach → 하강 → 진공 ON → 상승 → 놓기 → 진공 OFF+블로우" + (" → Home 복귀" if home else "  (⚠ Home 미저장 — 복귀 없음)"),
            "",
            "종료: 검출 0개(빈 비움) / 최대 반복 도달 / 정지 버튼 / 비상정지",
            "",
            "⚠ 로봇이 자율로 반복 동작합니다. ext_move 실행 중이어야 하며,",
            "⚠ 비상시 Space 또는 비상정지 버튼을 사용하세요.",
            "\n시작하시겠습니까?",
        ]
        if QMessageBox.question(self, "연속 픽 확인", "\n".join(msg), QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"속도 설정 실패:\n{e}")
            return

        self._auto_running = True
        self._auto_done = 0
        self.btn_auto_start.setEnabled(False)
        self.btn_auto_stop.setEnabled(True)
        self._auto_update_progress("시작")
        QTimer.singleShot(0, self._auto_step)  # UI 갱신 후 첫 사이클

    def _auto_stop(self, reason: str = "정지"):
        """연속 픽 종료 (현재 진행 중인 모션은 그대로 두고 반복만 멈춘다)."""
        was = self._auto_running
        self._auto_running = False
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_stop.setEnabled(False)
        self._auto_update_progress(reason)
        if was:
            self.main.statusBar().showMessage(f"⏹ 연속 픽 종료: {reason} (완료 {self._auto_done}회)")
            logger.info(f"연속 픽 종료: {reason} (완료 {self._auto_done}회)")

    def _auto_update_progress(self, state: str):
        self.auto_progress_label.setText(f"{self._auto_done}/{self.auto_max_spin.value()} · {state}")

    def _auto_step(self):
        """한 사이클: 캡처 → 검출 → 여는 방향 → 선택 → 픽 사이클 실행."""
        if not self._auto_running:
            return
        if self._auto_done >= self.auto_max_spin.value():
            self._auto_stop("최대 반복 도달")
            return

        n = self._auto_done + 1
        # ① 캡처
        self._auto_update_progress(f"{n}회차 캡처")
        self.main.statusBar().showMessage(f"[연속 픽 {n}] 캡처 중...")
        QApplication.processEvents()
        self._capture()
        if not self._auto_running:  # 캡처 중 정지 눌렀을 수 있음
            return
        if self.current_image is None:
            self._auto_stop("캡처 실패")
            return

        # ② SAM3 검출  ③ 여는 방향 (grasp 회전이 열림 방향을 쓸 수 있으므로 항상 계산)
        self._auto_update_progress(f"{n}회차 검출")
        QApplication.processEvents()
        self._detect_sam3()
        if not self._auto_running:
            return
        if not self.pick_objects:
            self._auto_stop("검출 없음 — 빈 비움 완료")
            return
        self._detect_opening()
        if not self._auto_running:
            return

        # ④ 자동 선택
        idx = self._auto_pick_index()
        if idx is None:
            self._auto_stop("파지 가능한 객체 없음 (Z 안전/자세 계산 실패)")
            return

        # ⑤ 선택 → target_pose 계산
        self._select_object(idx)
        if self.target_pose is None:
            self._auto_stop("선택 객체의 파지 자세 계산 실패")
            return

        # ⑥ 픽 사이클 (기존 상태머신 재사용) — 완료되면 다음 사이클로
        target = dict(self.target_pose)
        offset = self.approach_dist.value()
        dwell = self.vac_dwell_spin.value()
        place = getattr(self.main, "place_pose", None)
        home = self.main.home_pose
        ax, ay, az = self._compute_approach_position(target, offset)
        for z in (target["z"], az, place["z"]) + ((home["z"],) if home else ()):
            if z < self.z_min_spin.value():
                self._auto_stop(f"Z 안전 한계 초과 (z={z:.1f} < {self.z_min_spin.value():.1f})")
                return

        self._auto_update_progress(f"{n}회차 픽 실행")
        steps = self._build_pick_steps(target, offset, dwell, place, home, label=f"[{n}]")
        self._run_cycle(
            steps,
            f"연속 픽 {n}회차 완료",
            on_done=self._auto_after_pick,
            on_abort=lambda m: self._auto_stop(f"사이클 중단: {m}"),
        )

    def _auto_after_pick(self):
        """픽 사이클 완료 콜백 → 다음 사이클."""
        self._auto_done += 1
        if not self._auto_running:
            self._auto_stop("정지")
            return
        if self._auto_done >= self.auto_max_spin.value():
            self._auto_stop("최대 반복 도달")
            return
        self._auto_update_progress("다음 사이클 준비")
        QTimer.singleShot(400, self._auto_step)  # 진동 가라앉을 짧은 여유

    def _auto_pick_index(self) -> Optional[int]:
        """자동 선택 기준으로 픽할 객체의 검출 index 반환. 후보 없으면 None.

        후보마다 **실제 파지 TCP**(_compute_grasp_tcp, Grasp 설정 반영)를 구해
        로봇이 실제로 갈 지점 기준으로 정렬한다. Z 안전 한계를 못 넘는 후보는 제외.
        """
        if not self.pick_objects:
            return None
        cur_tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        if not cur_tcp:
            cur_tcp = {"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 180}
        z_min = self.z_min_spin.value()
        offset = self.approach_dist.value()

        cands = []
        for obj in self.pick_objects:
            i = obj["index"]
            det = self.detections[i] if i < len(self.detections) else {}
            tcp, err = self._compute_grasp_tcp(obj, det, cur_tcp)
            if tcp is None:
                continue
            _, _, az = self._compute_approach_position(tcp, offset)
            if tcp["z"] < z_min or az < z_min:
                continue  # Z 안전 한계 미달 → 후보 제외
            cands.append({"index": i, "tcp": tcp, "det": det})
        if not cands:
            return None

        crit = self.auto_crit_combo.currentData()
        if crit == "conf":
            cands.sort(key=lambda c: -float(c["det"].get("confidence", 0.0)))
        elif crit == "opening":
            cands.sort(key=lambda c: -float((c["det"].get("opening") or {}).get("confidence", 0.0)))
        elif crit == "center":
            if self.bin_box:
                bx, by = self.bin_box["cx"], self.bin_box["cy"]
            else:
                bx = float(np.mean([c["tcp"]["x"] for c in cands]))
                by = float(np.mean([c["tcp"]["y"] for c in cands]))
            cands.sort(key=lambda c: (c["tcp"]["x"] - bx) ** 2 + (c["tcp"]["y"] - by) ** 2)
        else:  # topmost (기본) — base Z 가 큰 것 = 더미 위쪽
            cands.sort(key=lambda c: -c["tcp"]["z"])

        best = cands[0]
        logger.info(f"연속 픽 선택({crit}): 객체 #{best['index'] + 1}, " f"TCP z={best['tcp']['z']:.1f}mm, 후보 {len(cands)}개")
        return best["index"]

    # ============================================================
    # Bin Box (작업 볼륨) — 충돌 방지 기반
    # ============================================================

    @staticmethod
    def _default_bin_box() -> Dict:
        """Bin Box 기본값 (base 좌표계, mm/deg)."""
        return {
            "cx": 0.0,
            "cy": 500.0,  # base XY 중심
            "sx": 300.0,
            "sy": 200.0,  # 가로·세로 크기
            "yaw": 0.0,  # base Z 축 회전(deg) — 빈이 축과 안 맞을 때
            "z_rim": 150.0,  # 빈 상단(림) 높이 — 진입 안전고 기준
            "z_floor": 0.0,  # 빈 바닥 높이 — 하강 하한
            "gripper_r": 30.0,  # 그리퍼 원기둥 근사 반경
            "wall_margin": 5.0,  # 벽 여유
            "floor_margin": 5.0,  # 바닥 여유
            "safe_height": 50.0,  # 림 위로 띄울 진입/이탈 안전고
        }

    @staticmethod
    def _bin_box_corners(bb: Dict, shrink_xy: float = 0.0) -> np.ndarray:
        """Bin Box → base 좌표 8 코너 (0~3 아랫면, 4~7 윗면, 같은 순서로 대응).

        shrink_xy > 0 이면 XY 를 그만큼 안쪽으로 축소 → **파지 허용 영역**
        (벽에서 그리퍼 반경+여유 만큼 들어온 상자).
        """
        hx = max(1e-3, bb["sx"] / 2.0 - shrink_xy)
        hy = max(1e-3, bb["sy"] / 2.0 - shrink_xy)
        th = np.radians(bb.get("yaw", 0.0))
        c, s = np.cos(th), np.sin(th)
        local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        out = []
        for z in (bb["z_floor"], bb["z_rim"]):
            for lx, ly in local:
                out.append([bb["cx"] + c * lx - s * ly, bb["cy"] + s * lx + c * ly, z])
        return np.asarray(out, dtype=float)

    def _base_to_cam_points(self, pts_base) -> Optional[np.ndarray]:
        """base 좌표 점들 (N,3) → 카메라 좌표계. 3D 뷰가 카메라 좌표계라 표시용 (CalibrationContext 위임)."""
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        pts = np.asarray(pts_base, dtype=float).reshape(-1, 3)
        return self.main.calib.base_to_cam_points(pts, tcp=tcp)

    def _render_bin_box(self):
        """3D 뷰에 Bin Box(노랑) + 파지 허용 영역(초록) 표시. 미설정이면 지움."""
        self.view_3d.remove_named("bin_box")
        self.view_3d.remove_named("bin_safe")
        if not self.bin_box:
            self.view_3d.plotter.render()
            return
        outer = self._base_to_cam_points(self._bin_box_corners(self.bin_box))
        if outer is None:
            self.main.statusBar().showMessage("Bin Box 표시 불가 — 캘리브레이션 로드 필요")
            return
        self.view_3d.remove_named("roi_box")  # ROI 자동 상자와 겹치지 않게 (Bin Box 가 기준)
        self.view_3d.show_wire_box(outer, "bin_box", "yellow", 4)
        shrink = self.bin_box["gripper_r"] + self.bin_box["wall_margin"]
        inner = self._base_to_cam_points(self._bin_box_corners(self.bin_box, shrink))
        if inner is not None:
            self.view_3d.show_wire_box(inner, "bin_safe", "lime", 3)
        self.view_3d.plotter.render()

    def _bin_box_from_drag(self) -> Optional[Dict]:
        """드래그한 ROI 영역 → Bin Box(base 좌표계). 불가하면 조용히 None.

        ROI 안 유효 3D 점을 base 로 변환한 뒤 **XY 평면 투영에 cv2.minAreaRect** 를 적용해
        중심·크기·yaw 를 얻는다 (빈이 비스듬해도 실제 방향을 잡아냄).

        **기존 Bin Box 가 있으면 z_rim/z_floor 와 그리퍼 파라미터는 유지**한다 —
        로봇으로 티칭한 높이가 XY 재조정 때문에 측정값으로 덮어써지면 안 되기 때문.
        처음 만들 때만 측정 점 분포(2%/98% 백분위)로 높이를 초기화한다.
        """
        if self.roi_2d is None or self.current_xyz is None or self.T_calib is None:
            return None
        h, w = self.current_xyz.shape[:2]
        x1, y1, x2, y2 = self.roi_2d
        x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        region = self.current_xyz[y1:y2, x1:x2].reshape(-1, 3)
        valid = region[~np.any(np.isnan(region), axis=1)]
        if len(valid) < 50:
            return None
        # 카메라 → base (모드 분기는 CalibrationContext 가 소유)
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        base_pts = self.main.calib.cam_to_base_points(valid, tcp=tcp)
        if base_pts is None:
            return None
        xy = base_pts[:, :2].astype(np.float32)
        (cx, cy), (sx, sy), ang = cv2.minAreaRect(xy.reshape(-1, 1, 2))
        if sx < 1 or sy < 1:
            return None

        if self.bin_box:  # 기존 값이 있으면 높이·그리퍼 설정은 그대로 둔다
            bb = dict(self.bin_box)
        else:  # 처음 생성 — 측정값으로 높이 초기화
            bb = self._default_bin_box()
            bb["z_floor"] = float(np.percentile(base_pts[:, 2], 2))
            bb["z_rim"] = float(np.percentile(base_pts[:, 2], 98))
        bb.update({"cx": float(cx), "cy": float(cy), "sx": float(sx), "sy": float(sy), "yaw": float(ang)})
        return bb

    def _bin_box_image_polygon(self) -> Optional[np.ndarray]:
        """Bin Box 림(상단) 4 코너 → 이미지 픽셀 폴리곤 (2D 표시 + roi_2d 산출용).

        base → 카메라 → intrinsics 핀홀 투영. 회전(yaw)이 그대로 반영된 사각형이 나온다.
        """
        if not self.bin_box or self.current_intrinsics is None:
            return None
        cam = self._base_to_cam_points(self._bin_box_corners(self.bin_box)[4:])  # 윗면 4점
        if cam is None:
            return None
        intr = np.asarray(self.current_intrinsics, dtype=float)
        fx, fy, cx_i, cy_i = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        pts = []
        for X, Y, Z in cam:
            if Z <= 1e-6:  # 카메라 뒤 → 투영 불가
                return None
            pts.append((fx * X / Z + cx_i, fy * Y / Z + cy_i))
        return np.asarray(pts, dtype=float)

    def _refresh_bin_box_views(self):
        """Bin Box 를 2D(투영 폴리곤) + 3D(와이어 상자) 양쪽에 표시하고 roi_2d 를 동기화.

        roi_2d 는 검출 ROI 필터(§3.1)의 1차 게이트라, 화면에 보이는 Bin Box 와
        필터 영역이 어긋나지 않도록 폴리곤의 바운딩 박스로 갱신한다.
        """
        if not self.bin_box:
            self.view_2d.set_roi_polygon(None)
            self._render_bin_box()
            return
        poly = self._bin_box_image_polygon()
        if poly is not None:
            self.view_2d.set_roi(None)  # 축 정렬 사각형 대신 폴리곤 표시
            self.view_2d.set_roi_polygon(poly)
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.roi_2d = (
                    float(np.clip(poly[:, 0].min(), 0, w)),
                    float(np.clip(poly[:, 1].min(), 0, h)),
                    float(np.clip(poly[:, 0].max(), 0, w)),
                    float(np.clip(poly[:, 1].max(), 0, h)),
                )
        self._render_bin_box()

    def _save_bin_box(self):
        try:
            with open(self._bin_box_path, "w") as f:
                json.dump(self.bin_box, f, indent=2)
        except Exception as e:
            logger.warning(f"Bin Box 저장 실패: {e}")

    def _load_bin_box(self):
        """앱 시작 시 자동 호출. 카메라가 고정이라 저장값이 계속 유효하다."""
        try:
            if os.path.exists(self._bin_box_path):
                with open(self._bin_box_path) as f:
                    loaded = json.load(f)
                bb = self._default_bin_box()
                bb.update(loaded)  # 새 필드가 늘어나도 안전
                self.bin_box = bb
                logger.info(f"Bin Box 로드: {self._bin_box_path}")
        except Exception as e:
            logger.warning(f"Bin Box 로드 실패: {e}")

    def _open_bin_box_config(self):
        get_tcp = (lambda: self.main.robot.get_tcp_position()) if self.main.robot else None
        dlg = BinBoxDialog(
            self.bin_box or self._default_bin_box(),
            get_current_tcp=get_tcp,
            parent=self,
        )
        if dlg.exec():
            self.bin_box = dlg.result_config()
            self._save_bin_box()
            self._refresh_bin_box_views()  # 2D 폴리곤 + 3D 상자 동시 갱신
            bb = self.bin_box
            self.main.statusBar().showMessage(
                f"Bin Box 설정: 중심({bb['cx']:.0f},{bb['cy']:.0f}) 크기({bb['sx']:.0f}×{bb['sy']:.0f}) "
                f"yaw {bb['yaw']:.1f}° Z[{bb['z_floor']:.0f}~{bb['z_rim']:.0f}]mm"
            )

    def _save_grasp_config(self):
        try:
            with open(self._grasp_cfg_path, "w") as f:
                json.dump(self.grasp_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Grasp 설정 저장 실패: {e}")

    def _load_grasp_config(self):
        """앱 시작 시 저장된 Grasp 설정 복원.

        Z 회전 제한 같은 **안전 설정이 재시작 때 기본값으로 돌아가면 위험**하므로
        (호스 감김) 파일로 유지한다. 새 항목이 추가돼도 안전하도록 기본값 위에 덮어쓴다.
        """
        try:
            if os.path.exists(self._grasp_cfg_path):
                with open(self._grasp_cfg_path) as f:
                    loaded = json.load(f)
                self.grasp_config.update(loaded)
                logger.info(f"Grasp 설정 로드: {self._grasp_cfg_path}")
        except Exception as e:
            logger.warning(f"Grasp 설정 로드 실패: {e}")

    def _open_grasp_config(self):
        get_tcp = (lambda: self.main.robot.get_tcp_position()) if self.main.robot else None
        dlg = GraspConfigDialog(self.grasp_config, get_current_tcp=get_tcp, parent=self)
        if dlg.exec():
            self.grasp_config = dlg.result_config()
            self._save_grasp_config()
            self.main.statusBar().showMessage("Grasp 설정 적용됨 (저장됨)")
            if self.selected_idx is not None:  # 선택 중이면 새 설정으로 즉시 재계산
                self._select_object(self.selected_idx)

    def _on_calibration_loaded(self):
        self._refresh_bin_box_views()  # base→cam 변환 가능해짐 → Bin Box 2D/3D 표시

    def _cam_to_base(self, center_cam, normal_cam):
        """카메라 좌표계 점/법선 → 로봇 base. (center_base, normal_base) 또는 (None,None).
        변환 로직은 CalibrationContext 가 소유 — eye_in_hand 에 필요한 현재 TCP 만 여기서 공급."""
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        return self.main.calib.cam_to_base(center_cam, normal_cam, tcp=tcp)

    def _pixel_to_base_on_plane(self, u, v, z_plane):
        """이미지 픽셀 (u,v) → 카메라 광선 → base 평면 Z=z_plane 과의 교점(base XYZ). 깊이 불사용.

        투명 객체는 픽셀 깊이가 불안정하므로, 알려진 작업 평면(z_plane)에 광선을 쏴
        XY(와 회전 방향)를 구한다. 실패 시 None."""
        intr = self.current_intrinsics
        if intr is None or self.T_calib is None:
            return None
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx_i, cy_i = float(intr[0, 2]), float(intr[1, 2])
        ray_cam = np.array([(u - cx_i) / fx, (v - cy_i) / fy, 1.0])
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        T = self.main.calib.T_cam_to_base(tcp)
        if T is None:
            return None
        O = T[:3, 3]
        d = T[:3, :3] @ ray_cam
        if abs(d[2]) < 1e-9:
            return None
        t = (z_plane - O[2]) / d[2]
        if t <= 0:
            return None
        return O + t * d

    def _grasp_pixel(self, obj, det, source):
        """파지 XY 로 쓸 이미지 픽셀 (u,v). obb_center/mask_center/bbox_center."""
        if source == "obb_center":
            mask = det.get("mask")
            obb = det.get("obb") or (oa.obb_from_mask(mask) if mask is not None else None)
            if obb is not None:
                det["obb"] = obb
                return tuple(obb["center"])
            # OBB 실패 시 마스크/ bbox 로 폴백
        if source != "bbox_center":
            mask = det.get("mask")
            if mask is not None:
                ys, xs = np.nonzero(np.asarray(mask))
                if len(xs):
                    return (float(xs.mean()), float(ys.mean()))
        bx = det.get("bbox")
        if bx is None:
            return None
        return ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)

    def _opening_dir_px(self, det, want):
        """이미지 좌표 단위벡터 (dx,dy). want='opening'(열림)|'obb_long'(장축). 실패 시 None."""
        mask = det.get("mask")
        obb = det.get("obb") or (oa.obb_from_mask(mask) if mask is not None else None)
        if obb is None:
            return None
        det["obb"] = obb
        if want == "obb_long":
            pts = np.asarray(obb["box_pts"], float)
            e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
            long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            n = np.linalg.norm(long_e)
            return (long_e / n) if n > 1e-6 else None
        # opening: det 에 있으면 재사용, 없으면 현재 UI 설정으로 계산
        op = det.get("opening")
        if op is None:
            if mask is None or self.current_rgb is None:
                return None
            gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_RGB2GRAY) if self.current_rgb.ndim == 3 else self.current_rgb
            weight = oa.opening_weight_map(gray, self.opening_method_combo.currentData(), self.opening_thr_spin.value())
            op = oa.opening_from_weight(mask, weight, obb, self.opening_erode_spin.value() / 100.0)
            if op is None:
                return None
            if self.opening_invert_chk.isChecked():
                op = dict(op)
                op["dir"] = (-op["dir"][0], -op["dir"][1])
            det["opening"] = op
        return tuple(op["dir"])

    def _yaw_from_direction(self, det, cfg):
        """이미지 방향(열림/OBB장축)을 작업 평면에 투영해 base yaw A(deg) 계산. 실패 시 None."""
        d_px = self._opening_dir_px(det, cfg["yaw_source"])
        obb = det.get("obb")
        if d_px is None or obb is None:
            return None
        cx, cy = obb["center"]
        L = 20.0
        p0 = self._pixel_to_base_on_plane(cx, cy, cfg["z_plane"])
        p1 = self._pixel_to_base_on_plane(cx + d_px[0] * L, cy + d_px[1] * L, cfg["z_plane"])
        if p0 is None or p1 is None:
            return None
        dxy = p1 - p0
        return float(np.degrees(np.arctan2(dxy[1], dxy[0])) + cfg["a_offset"])

    def _limit_yaw(self, a: float, cur_a: float) -> Optional[float]:
        """Z축 회전(A)을 허용 범위 안으로 되돌린다. 불가능하면 None (그 파지는 배제).

        **왜 필요한가:** 진공 호스가 로봇 팔을 따라 붙어 있어 손목이 한 방향으로 계속
        돌면 줄이 감긴다. 그런데 `compute_approach_pose` 의 ABC unwrap 은 "현재 자세에
        가장 가까운 표현"을 고르므로(모듈로-360), A 가 190°·200°… 로 **누적**될 수 있다.

        같은 물리 자세를 나타내는 후보들 중 범위 안에 들면서 현재 A 에 가장 가까운 것을 고른다:
          · A ± 360k  — 완전히 동일한 자세 (표현만 다름)
          · A ± 180   — 툴을 Z축으로 뒤집은 자세. 흡착 그리퍼는 패드가 원형이라 파지
                        지점이 같아 대개 동등하다. 다만 툴 X 방향이 반대가 되므로
                        '여는 방향 정렬'이 중요하면 yaw_allow_180 을 끈다.
        """
        cfg = self.grasp_config
        if not cfg.get("yaw_limit", True):
            return a
        lo, hi = float(cfg.get("yaw_min", -180.0)), float(cfg.get("yaw_max", 180.0))
        if lo > hi:
            lo, hi = hi, lo
        bases = [a, a + 180.0] if cfg.get("yaw_allow_180", False) else [a]
        cands = []
        for base in bases:
            for k in range(-3, 4):  # ±1080° 범위면 실용적으로 충분
                v = base + 360.0 * k
                if lo - 1e-9 <= v <= hi + 1e-9:
                    cands.append(v)
        if not cands:
            return None
        return min(cands, key=lambda v: abs(v - cur_a))  # 현재 자세에서 가장 덜 돌아가는 것

    def _compute_grasp_tcp(self, obj, det, cur_tcp):
        """grasp_config 에 따라 검출 객체의 로봇 base TCP 계산. (tcp dict, None) 또는 (None, 에러문구)."""
        cfg = self.grasp_config

        # ---- 위치 (x, y, z base) ----
        if cfg["pos_mode"] == "plane":
            px = self._grasp_pixel(obj, det, cfg["xy_source"])
            if px is None:
                return None, "위치 픽셀을 정할 수 없음 (마스크/bbox 필요)"
            base = self._pixel_to_base_on_plane(px[0], px[1], cfg["z_plane"])
            if base is None:
                return None, "평면 투영 실패 (intrinsics/캘리브레이션 확인)"
            x, y, z = float(base[0]), float(base[1]), float(cfg["z_pick"])
        else:  # cloud (기존)
            cb, _ = self._cam_to_base(np.array(obj["center"]), np.array(obj["normal"]))
            if cb is None:
                return None, "카메라→base 변환 실패"
            x, y, z = float(cb[0]), float(cb[1]), float(cb[2])

        # ---- 자세 (a, b, c) ----
        if cfg["approach"] == "normal":
            cb, nb = self._cam_to_base(np.array(obj["center"]), np.array(obj["normal"]))
            if nb is None:
                return None, "카메라→base 변환 실패"
            from calibration import compute_approach_pose

            ap = compute_approach_pose(cb, nb, cur_tcp)
            a, b, c = ap["a"], ap["b"], ap["c"]
            if cfg["pos_mode"] == "cloud":  # 법선+3D = 완전 기존 동작
                x, y, z = ap["x"], ap["y"], ap["z"]
        else:  # vertical (수직 top-down)
            b, c = float(cfg["b_fixed"]), float(cfg["c_fixed"])
            if cfg["yaw_source"] == "fixed":
                a = float(cfg["a_fixed"])
            else:
                a = self._yaw_from_direction(det, cfg)
                if a is None:
                    return None, "회전(yaw) 방향을 정할 수 없음 (마스크/OBB/열림 방향 필요)"

        # Z축 회전 제한 (호스/케이블 감김 방지) — 범위를 못 맞추면 이 파지는 배제
        a_limited = self._limit_yaw(float(a), float(cur_tcp.get("a", 0.0)))
        if a_limited is None:
            cfg_y = self.grasp_config
            return None, (
                f"Z 회전 제한 초과 (A={a:.1f}° → 허용 " f"{cfg_y.get('yaw_min', -180):.0f}~{cfg_y.get('yaw_max', 180):.0f}° 안으로 못 맞춤)"
            )
        a = a_limited

        return {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c}, None

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
        # cam→base 회전의 전치 = base→cam (모드 분기는 CalibrationContext 소유)
        tcp = self.main.robot.get_tcp_position() if self.main.robot else None
        T_c2b = self.main.calib.T_cam_to_base(tcp)
        if T_c2b is not None:
            R_in_cam = T_c2b[:3, :3].T @ R_target_base
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
                line,
                color=color,
                line_width=6,
                name=name,
                render_lines_as_tubes=True,
                pickable=False,
                reset_camera=False,
            )
            self._tcp_viz_actors.append(name)

        # Approach 지점 = Tool -Z 방향으로 offset 떨어진 곳 (카메라 좌표계 -Z)
        offset = float(self.approach_dist.value()) if self.use_approach.isChecked() else 50.0
        approach_pos = (origin - R_in_cam[:, 2] * offset).astype(np.float32)
        sphere = pv.Sphere(radius=4, center=approach_pos)
        plotter.add_mesh(
            sphere,
            color="#ffaa00",
            name="tcp_approach",
            pickable=False,
            reset_camera=False,
        )
        self._tcp_viz_actors.append("tcp_approach")

        # Approach → Target 경로선
        path = pv.PolyData(np.array([approach_pos, origin], dtype=np.float32))
        path.lines = np.array([2, 0, 1])
        plotter.add_mesh(
            path,
            color="#ffaa00",
            line_width=3,
            name="tcp_path",
            render_lines_as_tubes=True,
            pickable=False,
            reset_camera=False,
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
