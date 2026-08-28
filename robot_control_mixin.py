"""
RobotControlMixin — BinPickingTab과 CADMatchingTab이 공유하는 로봇 제어 /
시퀀스 큐 / 안전 기능 메서드 모음.

두 탭에서 중복되던 19개 메서드를 이 믹스인으로 추출했다. 동작은 완전히 동일하며
(객체/인스턴스 명사 차이만 SEQ_OBJECT_NOUN 클래스 속성으로 매개변수화),
각 탭은 `class XxxTab(RobotControlMixin, QWidget)` 형태로 상속한다.

동작뿐 아니라 **레이아웃도** 여기서 소유한다. 탭들이 똑같은 패널 코드를 복붙하는 바람에
버튼 하나 옮기려면 탭마다 손을 대야 했고 배치가 조금씩 어긋났기 때문에, 다음 빌더로 합쳤다:
  _build_move_group()  — `로봇 이동 제어` 그룹박스 통째로 (빈 픽킹 / CAD 매칭)
  _build_seq_group()   — `시퀀스 큐` 그룹박스 통째로 (빈 픽킹 / CAD 매칭)
  _build_home_row() / _build_place_row() / _build_safety_rows() / _build_vacuum_row()
                       — 패널 전체는 안 맞고 일부 행만 필요한 탭(표면 추적)이 골라 쓰는 하위 빌더
이 빌더들이 믹스인의 나머지 메서드가 참조하는 위젯을 만든다: speed_spin, z_min_spin,
move_mode_combo, use_approach, approach_dist, action_list, btn_move, btn_move_home,
btn_set_home, btn_move_place, btn_set_place, btn_add_obj_to_seq, btn_add_home_to_seq …

빌더가 만들지 않아 **탭이 직접 준비해야 하는** 것: self.main (.robot, .home_pose,
.statusBar()), self.target_pose, self.selected_idx, self.user_queue, self._current_mode.
또한 각 탭 고유의 self._refresh_mode_display() 와 self._execute_move() 를 호출한다
(MRO상 탭 구현이 사용됨).
"""

import time
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np  # noqa: F401  (탭 코드와의 일관성 위해 유지)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QGroupBox,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QListWidget,
)

from calibration import tcp_to_homogeneous
from kuka_robot import is_auto_mode

logger = logging.getLogger(__name__)


class RobotControlMixin:
    # 시퀀스 라벨/메시지에서 사용하는 대상 명사.
    # BinPickingTab은 기본값("객체"), CADMatchingTab은 "인스턴스"로 오버라이드.
    SEQ_OBJECT_NOUN = "객체"

    # ============================================================
    # AUT 모드 안전 기능
    # ============================================================

    def _is_aut_mode(self) -> bool:
        return is_auto_mode(self._current_mode)

    def _effective_speed(self, requested: int) -> int:
        if self._is_aut_mode():
            return min(requested, 50)
        return requested

    def _validate_z(self, z: float) -> bool:
        z_min = self.z_min_spin.value()
        if z < z_min:
            QMessageBox.critical(
                self,
                "Z 안전 한계 초과",
                f"목표 Z={z:.1f}mm가 한계 {z_min:.1f}mm보다 낮습니다.\n이동을 차단합니다.",
            )
            return False
        return True

    def _emergency_stop(self):
        """비상정지 (robo_scram=TRUE → KRL 즉시 brake). 두 탭 동일 동작."""
        if self.main.robot is None:
            self.main.statusBar().showMessage("로봇 미연결 - 비상정지 무시")
            return
        try:
            self.main.robot.emergency_stop()
            self.main.statusBar().showMessage("⛔ 비상정지 트리거됨 (해제 버튼 누르면 재시작 가능)")
            logger.warning("비상정지 트리거")
        except Exception as e:
            logger.error(f"비상정지 오류: {e}")
        # 실행 중인 픽/시퀀스 사이클도 중단 (남은 스텝 전송 방지)
        if self._cycle_is_running():
            self._cycle_abort("비상정지로 사이클 중단")
        # 연속 픽(자동 반복) 루프도 즉시 종료 — 사이클이 안 돌고 있던 순간
        # (캡처/검출 중)에 눌렸을 수도 있으므로 별도로 확인한다.
        if getattr(self, "_auto_running", False):
            self._auto_stop("비상정지")

    def _emergency_stop_release(self):
        """
        비상정지 해제 (robo_scram=FALSE만). 큐는 그대로 유지.

        안전성: KRL의 robo_scram_DEF가 RESUME으로 현재 진행 중이던 모션을
        자동 취소하므로, 해제 직후 멈췄던 모션이 그대로 재개되지는 않는다.
        큐의 다음 슬롯은 정상 흐름으로 실행됨 — 큐를 비우고 싶으면 별도의
        '큐 비우기' 버튼을 사용 (UI 일관성 + 사용자 선택권).
        """
        if self.main.robot is None:
            return
        try:
            self.main.robot.emergency_stop_release()
            self.main.statusBar().showMessage("비상정지 해제됨 (큐는 유지 - 비우려면 '큐 비우기' 버튼)")
            logger.info("비상정지 해제")
        except Exception as e:
            logger.error(f"비상정지 해제 오류: {e}")

    def _on_robot_connected(self):
        """로봇 연결 시 main이 호출. Home/진공 관련 버튼 활성화."""
        self.btn_set_home.setEnabled(True)
        if self.main.home_pose:
            self.btn_move_home.setEnabled(True)
            self.btn_add_home_to_seq.setEnabled(True)
        # 진공/픽 버튼 (탭이 _build_vacuum_row() 를 호출한 경우에만 존재)
        for name in (
            "btn_vac_on",
            "btn_vac_off",
            "btn_vac_blow",
            "btn_pick_cycle",
            "btn_set_place",
            "btn_move_place",
            "btn_add_pick_to_seq",
            "btn_auto_start",
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(True)
        self._refresh_mode_display()

    # ============================================================
    # 공용 UI 빌더 — "로봇 이동 제어" / "시퀀스 큐" 패널
    # ============================================================
    #
    # 예전에는 이 두 패널을 각 탭의 _init_ui 가 **똑같은 코드로 각각** 만들었다.
    # 그래서 버튼 하나를 옮기려면 탭마다 손으로 고쳐야 했고, 실제로 배치가
    # 조금씩 어긋나 있었다. 레이아웃도 동작처럼 여기 한 곳에 모아서,
    # **한 번 고치면 이 빌더를 쓰는 모든 탭에 동시에 반영**되게 한다.
    #
    # 사용법 (탭의 _init_ui):
    #     info_layout.addWidget(self._build_move_group())
    #     info_layout.addWidget(self._build_seq_group())
    #
    # 패널 통째로는 안 맞고 일부 행만 필요한 탭(표면 추적 = "실행 제어" 그룹)은
    # 하위 빌더(_build_home_row / _build_place_row / _build_safety_rows)만 골라 쓴다.

    # 이동 방식 콤보 항목 (인덱스 0 = LIN 이 기본).
    MOVE_MODE_ITEMS = ["LIN (직선, 추천)", "PTP (최단 경로)"]
    # 속도(%) 초기값 — AUT 모드에서는 _effective_speed 가 50% 로 한 번 더 제한한다.
    MOVE_SPEED_DEFAULT = 30
    # Z 최소 한계(mm) 초기값 — 바닥 충돌 방지.
    MOVE_Z_MIN_DEFAULT = 5

    def _build_move_group(self) -> QGroupBox:
        """`로봇 이동 제어` 그룹박스를 통째로 생성해서 돌려준다.

        구성 (위 → 아래):
            이동 방식 / 속도(%) / 접근·철수 / Z 최소(mm) / 흡착대기(s)
            → 선택 위치로 이동
            → [Home 으로 이동 | Home 재설정]
            → [놓기 위치로 이동 | 놓기 위치 재설정]
            → 진공 그리퍼 행 + 픽 실행
            → 큐 비우기 / 비상정지 / 비상정지 해제

        여기서 만드는 위젯은 믹스인의 다른 메서드들이 이름으로 참조한다
        (speed_spin, z_min_spin, move_mode_combo, use_approach, approach_dist,
        btn_move, btn_move_home, btn_set_home, btn_move_place, btn_set_place …).
        """
        group = QGroupBox("로봇 이동 제어")
        layout = QVBoxLayout(group)

        # --- 이동 방식 (LIN = 직선 보간 / PTP = 관절 최단 경로) ---
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("방식:"))
        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(self.MOVE_MODE_ITEMS)
        mode_row.addWidget(self.move_mode_combo)
        layout.addLayout(mode_row)

        # --- 속도(%) — 값을 바꾸면 즉시 로봇에 적용($OV_PRO) ---
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("속도(%):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(self.MOVE_SPEED_DEFAULT)
        self.speed_spin.setFixedWidth(70)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_spin)
        self.btn_apply_speed = QPushButton("적용")
        self.btn_apply_speed.setFixedWidth(50)
        self.btn_apply_speed.clicked.connect(self._apply_speed_now)
        speed_row.addWidget(self.btn_apply_speed)
        speed_row.addStretch()
        layout.addLayout(speed_row)

        # --- 접근/철수 (Approach → Target → Retract 3단계) ---
        approach_row = QHBoxLayout()
        self.use_approach = QCheckBox("접근/철수 사용")
        self.use_approach.setChecked(True)
        self.use_approach.setToolTip(
            "체크 시 [Approach → Target → Retract] 3단계 모션을 큐에 추가\n" "Tool +Z 방향으로 위에 안전하게 다가갔다 → 정밀 접근 → 다시 위로"
        )
        approach_row.addWidget(self.use_approach)
        approach_row.addWidget(QLabel("거리(mm):"))
        self.approach_dist = QSpinBox()
        self.approach_dist.setRange(5, 500)
        self.approach_dist.setValue(50)
        self.approach_dist.setFixedWidth(60)
        approach_row.addWidget(self.approach_dist)
        approach_row.addStretch()
        layout.addLayout(approach_row)

        # --- Z 최소 한계 (안전: 바닥 충돌 방지) ---
        zlim_row = QHBoxLayout()
        zlim_row.addWidget(QLabel("Z 최소(mm):"))
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(-2000, 2000)
        self.z_min_spin.setValue(self.MOVE_Z_MIN_DEFAULT)
        self.z_min_spin.setFixedWidth(80)
        self.z_min_spin.setToolTip("타겟 Z 좌표가 이 값보다 낮으면 이동을 거부합니다 (바닥 충돌 방지)")
        zlim_row.addWidget(self.z_min_spin)
        zlim_row.addStretch()
        layout.addLayout(zlim_row)

        # --- 흡착대기(s) — Z 최소와 같은 "설정" 성격이라 바로 아래 배치 ---
        dwell_row = QHBoxLayout()
        dwell_row.addWidget(QLabel("흡착대기(s):"))
        dwell_row.addWidget(self._make_dwell_spin())
        dwell_row.addStretch()
        layout.addLayout(dwell_row)
        # 이 그룹이 흡착대기·놓기 버튼을 자체 행으로 만들었으므로, 아래
        # _build_vacuum_row() 가 같은 위젯을 또 만들지 않게 인스턴스 속성으로 덮어쓴다.
        self.DWELL_SPIN_IN_OWN_ROW = True
        self.PLACE_BUTTONS_IN_OWN_ROW = True

        # --- 선택 위치로 이동 (큰 파랑) ---
        self.btn_move = QPushButton("선택 위치로 이동")
        self.btn_move.setMinimumHeight(45)
        self.btn_move.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1976D2; color: white;")
        self.btn_move.clicked.connect(self._execute_move)
        self.btn_move.setEnabled(False)
        layout.addWidget(self.btn_move)

        layout.addLayout(self._build_home_row())
        layout.addLayout(self._build_place_row())
        layout.addLayout(self._build_vacuum_row())
        self._build_safety_rows(layout)
        return group

    def _build_home_row(self) -> QHBoxLayout:
        """[🏠 Home 으로 이동 | 📍 Home 재설정] 한 줄 (2:1 비율)."""
        row = QHBoxLayout()

        self.btn_move_home = QPushButton("🏠 Home으로 이동")
        self.btn_move_home.setMinimumHeight(40)
        self.btn_move_home.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #2E7D32; color: white;")
        self.btn_move_home.setToolTip("저장된 Home 위치로 PTP 이동합니다")
        self.btn_move_home.clicked.connect(self._move_to_home)
        self.btn_move_home.setEnabled(False)
        row.addWidget(self.btn_move_home, stretch=2)

        self.btn_set_home = QPushButton("📍 Home\n재설정")
        self.btn_set_home.setMinimumHeight(40)
        self.btn_set_home.setStyleSheet("font-size: 11px; background-color: #689F38; color: white;")
        self.btn_set_home.setToolTip("현재 로봇 TCP 위치를 새 Home으로 저장합니다")
        self.btn_set_home.clicked.connect(self._set_home_to_current)
        self.btn_set_home.setEnabled(False)
        row.addWidget(self.btn_set_home, stretch=1)
        return row

    def _build_place_row(self) -> QHBoxLayout:
        """[📦 놓기 위치로 이동 | 📍 놓기 위치 재설정] 한 줄 — Home 행과 같은 구성/크기."""
        row = QHBoxLayout()

        self.btn_move_place = QPushButton("📦 놓기 위치로 이동")
        self.btn_move_place.setMinimumHeight(40)
        self.btn_move_place.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #00695C; color: white;")
        self.btn_move_place.setToolTip("저장된 놓기(Place) 위치로 PTP 이동합니다")
        self.btn_move_place.clicked.connect(self._move_to_place)
        self.btn_move_place.setEnabled(False)
        row.addWidget(self.btn_move_place, stretch=2)

        self.btn_set_place = QPushButton("📍 놓기 위치\n재설정")
        self.btn_set_place.setMinimumHeight(40)
        self.btn_set_place.setStyleSheet("font-size: 11px; background-color: #26A69A; color: white;")
        self.btn_set_place.setToolTip("현재 로봇 TCP 위치를 '놓기(Place) 위치'로 저장 — 픽 사이클이 여기에 내려놓음")
        self.btn_set_place.clicked.connect(self._set_place_to_current)
        self.btn_set_place.setEnabled(False)
        row.addWidget(self.btn_set_place, stretch=1)
        return row

    def _build_safety_rows(self, layout: QVBoxLayout) -> None:
        """큐 비우기 / 비상정지 / 비상정지 해제 — 항상 패널 맨 아래에 붙는 3개 버튼."""
        self.btn_clear_queue = QPushButton("🗑 큐 비우기 (이전 명령 취소)")
        self.btn_clear_queue.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold;")
        self.btn_clear_queue.setToolTip("KRL 모션 큐에 남아 있는 이전 명령을 모두 취소합니다")
        self.btn_clear_queue.clicked.connect(self._clear_motion_queue)
        layout.addWidget(self.btn_clear_queue)

        self.btn_estop = QPushButton("⛔ 비상정지 (Space)")
        self.btn_estop.setMinimumHeight(60)
        self.btn_estop.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #D32F2F; color: white;")
        self.btn_estop.clicked.connect(self._emergency_stop)
        layout.addWidget(self.btn_estop)

        self.btn_estop_release = QPushButton("비상정지 해제")
        self.btn_estop_release.setStyleSheet("background-color: #757575; color: white;")
        self.btn_estop_release.clicked.connect(self._emergency_stop_release)
        layout.addWidget(self.btn_estop_release)

    def _build_seq_group(self) -> QGroupBox:
        """`시퀀스 큐` 그룹박스를 통째로 생성해서 돌려준다.

        시퀀스 큐 = 사용자가 Python 쪽에서 미리 쌓아두는 액션 시나리오
        (객체 이동 / Home / 픽). `▶ 시퀀스 시작` 을 누르면 위에서부터 차례로
        로봇 모션 큐에 흘려보낸다 — 실행 로직은 _start_sequence 이하 참고.
        """
        group = QGroupBox("시퀀스 큐 (자동 실행 순서)")
        layout = QVBoxLayout(group)

        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(80)
        self.action_list.setMaximumHeight(150)
        layout.addWidget(self.action_list)

        # --- 추가 버튼들 ---
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

        add_row.addWidget(self._make_add_pick_to_seq_button())
        layout.addLayout(add_row)

        # --- 제거 버튼들 ---
        del_row = QHBoxLayout()
        self.btn_remove_seq_item = QPushButton("선택 항목 제거")
        self.btn_remove_seq_item.clicked.connect(self._remove_selected_action)
        del_row.addWidget(self.btn_remove_seq_item)

        self.btn_clear_seq = QPushButton("시퀀스 비우기")
        self.btn_clear_seq.clicked.connect(self._clear_user_queue)
        del_row.addWidget(self.btn_clear_seq)
        layout.addLayout(del_row)

        # --- 시작 버튼 (큰 파랑) ---
        self.btn_start_seq = QPushButton("▶ 시퀀스 시작")
        self.btn_start_seq.setMinimumHeight(45)
        self.btn_start_seq.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1565C0; color: white;")
        self.btn_start_seq.clicked.connect(self._start_sequence)
        layout.addWidget(self.btn_start_seq)
        return group

    # ============================================================
    # 진공 그리퍼 (SMC ZK2 — $OUT[7]=VAC_ON / $OUT[8]=VAC_Blow)
    # ============================================================

    # 놓기 위치 버튼 / 흡착대기 스핀이 **이미 다른 행에 만들어졌는지** 표시하는 플래그.
    # True 면 아래 _build_vacuum_row() 가 같은 위젯을 또 만들지 않는다 (중복 방지).
    # _build_move_group() 을 쓰는 탭(빈 픽킹·CAD 매칭)은 그 안에서 자동으로 True 가 되고,
    # 진공 행만 따로 가져다 쓰는 탭(표면 추적)은 기본값 False 라 예전처럼 한 줄에 모인다.
    PLACE_BUTTONS_IN_OWN_ROW = False
    DWELL_SPIN_IN_OWN_ROW = False

    def _build_vacuum_row(self) -> QVBoxLayout:
        """진공 그리퍼 + 픽 사이클 UI 생성 (2행). 탭의 _init_ui 에서 호출해 레이아웃에 추가.

        1행: 수동 진공 ON/OFF/블로우 (테스트/티칭용 — 상용 프로그램의 수동 I/O 패널 역할)
        2행: 픽 사이클 실행 / 놓기 위치 저장 / 흡착 대기 / 시퀀스에 픽 추가

        실행 조건: ext_move 가 실행 중(#P_ACTIVE)이어야 KRL 인터럽트가 $OUT 에
        적용한다 (kuka_robot.set_vacuum 이 readback 으로 실제 적용을 검증).
        """
        vbox = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_vac_on = QPushButton("🔵 진공 ON (잡기)")
        self.btn_vac_on.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold;")
        self.btn_vac_on.clicked.connect(lambda: self._set_vacuum_ui(True))
        self.btn_vac_on.setEnabled(False)
        row.addWidget(self.btn_vac_on)

        self.btn_vac_off = QPushButton("⚪ 진공 OFF (놓기)")
        self.btn_vac_off.setStyleSheet("background-color: #546E7A; color: white; font-weight: bold;")
        self.btn_vac_off.clicked.connect(lambda: self._set_vacuum_ui(False))
        self.btn_vac_off.setEnabled(False)
        row.addWidget(self.btn_vac_off)

        self.btn_vac_blow = QPushButton("💨 블로우")
        self.btn_vac_blow.setToolTip("0.5초 블로우 펄스 — 진공 OFF 후 물체를 확실히 떨어뜨릴 때")
        self.btn_vac_blow.clicked.connect(self._vacuum_blow_ui)
        self.btn_vac_blow.setEnabled(False)
        row.addWidget(self.btn_vac_blow)
        vbox.addLayout(row)

        # --- 2행: 픽 사이클 (상용 스타일 자동 픽: Approach→하강→진공→상승→놓기) ---
        pick_row = QHBoxLayout()
        self.btn_pick_cycle = QPushButton("🤖 픽 실행 (잡기→놓기)")
        self.btn_pick_cycle.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        self.btn_pick_cycle.setToolTip(
            "선택된 대상을 자동 픽: Approach(PTP) → 하강(LIN) → 진공 ON → 대기 →\n"
            "상승(LIN) → 놓기 위치로 이동 → 진공 OFF+블로우 → Home 복귀(저장돼 있으면)"
        )
        self.btn_pick_cycle.clicked.connect(self._execute_pick_cycle)
        self.btn_pick_cycle.setEnabled(False)
        # stretch=1 — 흡착대기·놓기 버튼을 다른 곳으로 옮긴 탭에서는 이 버튼이
        # 남는 가로 공간을 채운다 (빈 공간이 생기지 않게).
        pick_row.addWidget(self.btn_pick_cycle, stretch=1)

        if not self.DWELL_SPIN_IN_OWN_ROW:
            # 이 탭은 흡착대기를 따로 배치하지 않으므로 여기에 만든다
            pick_row.addWidget(QLabel("흡착대기(s)"))
            pick_row.addWidget(self._make_dwell_spin())

        if not self.PLACE_BUTTONS_IN_OWN_ROW:
            # 이 탭은 놓기 버튼을 따로 배치하지 않으므로 여기에 만든다
            self.btn_set_place = QPushButton("📍 놓기 위치 저장")
            self.btn_set_place.setToolTip("현재 로봇 TCP 위치를 '놓기(Place) 위치'로 저장 — 픽 사이클이 여기에 내려놓음")
            self.btn_set_place.clicked.connect(self._set_place_to_current)
            self.btn_set_place.setEnabled(False)
            pick_row.addWidget(self.btn_set_place)

        vbox.addLayout(pick_row)
        return vbox

    def _make_dwell_spin(self) -> QDoubleSpinBox:
        """흡착대기(s) 스핀 생성. 탭이 원하는 위치에 직접 배치할 수 있도록 분리했다."""
        self.vac_dwell_spin = QDoubleSpinBox()
        self.vac_dwell_spin.setRange(0.1, 5.0)
        self.vac_dwell_spin.setSingleStep(0.1)
        self.vac_dwell_spin.setValue(0.5)
        self.vac_dwell_spin.setFixedWidth(80)
        self.vac_dwell_spin.setToolTip("진공 ON 후 상승 전 대기 시간 (흡착이 자리잡을 시간)")
        return self.vac_dwell_spin

    def _make_add_pick_to_seq_button(self) -> QPushButton:
        """'픽(잡기→놓기)을 시퀀스 큐에 추가' 버튼 생성. 시퀀스 큐 그룹의 추가 버튼 행에
        '객체 이동 추가'/'Home 추가' 와 나란히 배치하려고 탭이 호출한다."""
        self.btn_add_pick_to_seq = QPushButton("➕ 픽 추가")
        self.btn_add_pick_to_seq.setStyleSheet("background-color: #8E24AA; color: white;")
        self.btn_add_pick_to_seq.setToolTip("선택된 대상의 '픽(잡기→놓기)' 액션을 시퀀스 큐에 추가")
        self.btn_add_pick_to_seq.clicked.connect(self._enqueue_pick)
        self.btn_add_pick_to_seq.setEnabled(False)
        return self.btn_add_pick_to_seq

    def _set_vacuum_ui(self, on: bool):
        if self.main.robot is None:
            self.main.statusBar().showMessage("로봇 미연결 - 진공 제어 무시")
            return
        label = "ON (잡기)" if on else "OFF (놓기)"
        self.main.statusBar().showMessage(f"진공 {label} 적용 중...")
        try:
            ok = self.main.robot.set_vacuum(on)
        except Exception as e:
            logger.error(f"진공 제어 오류: {e}")
            ok = False
        if ok:
            self.main.statusBar().showMessage(f"진공 {label} ✓")
        else:
            QMessageBox.warning(
                self,
                "진공 제어 실패",
                "출력이 적용되지 않았습니다.\n"
                "SmartPad에서 ext_move 가 실행 중(#P_ACTIVE)인지 확인하세요.\n"
                "(T1 모드는 Start 를 누르고 있어야 프로그램이 돕니다)",
            )
            self.main.statusBar().showMessage(f"진공 {label} 실패 — ext_move 실행 상태 확인")

    def _vacuum_blow_ui(self):
        if self.main.robot is None:
            self.main.statusBar().showMessage("로봇 미연결 - 블로우 무시")
            return
        self.main.statusBar().showMessage("블로우 펄스 (0.5s)...")
        try:
            ok = self.main.robot.vacuum_blow(0.5)
        except Exception as e:
            logger.error(f"블로우 오류: {e}")
            ok = False
        self.main.statusBar().showMessage("블로우 완료 ✓" if ok else "블로우 실패 — ext_move 실행 상태 확인")

    def _set_place_to_current(self):
        """현재 TCP 를 놓기(Place) 위치로 저장 (Home 재설정과 동일한 확인창 UX)."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        cur = self.main.robot.get_tcp_position()
        cur_axis = self.main.robot.get_axis_position()  # 관절값도 티칭 — 놓기 이동을 관절 PTP 로 (±180° A 부호 문제 제거)
        if cur is None:
            QMessageBox.critical(self, "오류", "현재 TCP 위치를 읽지 못했습니다")
            return

        old = getattr(self.main, "place_pose", None)
        old_str = (
            f"  X: {old['x']:.2f}, Y: {old['y']:.2f}, Z: {old['z']:.2f}\n" f"  A: {old['a']:.2f}, B: {old['b']:.2f}, C: {old['c']:.2f}"
            if old
            else "(저장된 놓기 위치 없음)"
        )
        new_str = f"  X: {cur['x']:.2f}, Y: {cur['y']:.2f}, Z: {cur['z']:.2f}\n" f"  A: {cur['a']:.2f}, B: {cur['b']:.2f}, C: {cur['c']:.2f}"
        msg = (
            f"📍 현재 위치를 놓기(Place) 위치로 저장하시겠습니까?\n\n"
            f"[기존 놓기 위치]\n{old_str}\n\n[새 놓기 위치 (현재 TCP)]\n{new_str}\n\n"
            f"픽 사이클이 물체를 이 위치에 내려놓습니다."
        )
        ret = QMessageBox.question(self, "놓기 위치 저장 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        self.main.place_pose = cur
        self.main.place_axis = dict(cur_axis) if cur_axis else None
        if cur_axis is None:
            logger.warning("관절값($AXIS_ACT)을 읽지 못해 놓기 위치는 Cartesian 으로만 저장됨 (관절 복귀 비활성)")
        self.main._save_teach_poses()  # 앱을 껐다 켜도 유지되도록 teach_poses.json 에 기록
        QMessageBox.information(
            self,
            "저장 완료",
            f"놓기 위치가 저장되었습니다.\n\n{new_str}",
        )
        self.main.statusBar().showMessage(f"📍 놓기 위치 저장됨: X={cur['x']:.1f}, Y={cur['y']:.1f}, Z={cur['z']:.1f}")
        logger.info(f"Place 위치 저장: {cur}")

    # ============================================================
    # 픽 사이클 — 단계별 실행기 (QTimer 상태머신)
    #
    # 상용 빈피킹의 표준 사이클을 구현: 모션은 KRL 큐에 "한 슬롯씩" 보내고
    # 완료(robo_motion_type[slot]==0)를 폴링으로 기다린 뒤, 사이 스텝에서
    # 진공 ON/OFF 를 삽입한다. (모든 모션을 한꺼번에 던지는 기존 방식으로는
    # 그리퍼 동작을 끼워넣을 수 없음)
    #
    # 스텝 형식: ("status", 문구) / ("move", "ptp"|"lin", pose_dict) / ("move_axis", axis_dict)
    #            / ("vacuum", bool) / ("blow", 초) / ("dwell", 초)
    # ============================================================

    def _cycle_is_running(self) -> bool:
        return bool(getattr(self, "_cycle_active", False))

    def _run_cycle(self, steps: List[Tuple], done_msg: str = "사이클 완료", on_done=None, on_abort=None):
        if self._cycle_is_running():
            QMessageBox.warning(self, "실행 중", "이미 사이클이 실행 중입니다")
            return
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        self._cycle_steps = list(steps)
        self._cycle_idx = 0
        self._cycle_pending_slots: List[int] = []
        self._cycle_dwell_until = 0.0
        self._cycle_done_msg = done_msg
        # 완료/중단 콜백 — 연속 픽 루프가 다음 사이클을 이어가거나 즉시 멈추는 데 쓴다.
        self._cycle_on_done = on_done
        self._cycle_on_abort = on_abort
        self._cycle_active = True
        if getattr(self, "_cycle_timer", None) is None:
            self._cycle_timer = QTimer(self)
            self._cycle_timer.timeout.connect(self._cycle_tick)
        self._cycle_timer.start(80)
        logger.info(f"사이클 시작: {len(steps)} 스텝")

    def _cycle_abort(self, msg: str):
        timer = getattr(self, "_cycle_timer", None)
        if timer is not None:
            timer.stop()
        self._cycle_active = False
        self._cycle_pending_slots = []
        self.main.statusBar().showMessage(f"⚠ {msg}")
        logger.warning(f"사이클 중단: {msg}")
        # 중단됐으므로 이어가기 콜백은 버리고, 중단 콜백만 알린다
        # (비상정지·오류 시 연속 픽 루프가 계속 돌면 안 됨)
        cb = getattr(self, "_cycle_on_abort", None)
        self._cycle_on_done = None
        self._cycle_on_abort = None
        if cb is not None:
            cb(msg)

    def _cycle_tick(self):
        """사이클 상태 머신 1틱.

        핵심 안전 규칙: 진공/블로우/dwell 스텝은 **먼저 보낸 모든 모션이
        물리적으로 완료된 뒤**에만 실행된다 (pending 슬롯이 빌 때까지 대기).
        KRL 쪽 WAIT SEC 0 덕분에 robo_motion_type[slot]==0 이 "물리적 도착"을
        보장한다 — 이 두 가지가 합쳐져야 이동 중 진공 OFF 같은 사고가 없다.

        효율: 연속된 모션 스텝은 KRL 큐에 한꺼번에 배치 전송해 모션 사이
        폴링 데드타임을 없애고, status 같은 즉시 스텝도 한 틱에 몰아 처리한다.
        """
        robot = self.main.robot
        if robot is None:
            self._cycle_abort("로봇 연결 끊김 — 사이클 중단")
            return

        try:
            # 1) 보낸 모션들의 물리적 완료 확인 (순차 실행이므로 앞에서부터 0이 됨)
            while self._cycle_pending_slots:
                val = robot.read_variable(f"robo_motion_type[{self._cycle_pending_slots[0]}]")
                if val is None or val.strip() != "0":
                    return  # 아직 이동 중 — 다음 틱에 재확인
                self._cycle_pending_slots.pop(0)

            # 2) dwell(대기) 진행 중이면 기다림
            if self._cycle_dwell_until:
                if time.time() < self._cycle_dwell_until:
                    return
                self._cycle_dwell_until = 0.0

            # 3) 스텝 소진 루프 — 즉시 스텝은 한 틱에 몰아서, 연속 모션은 배치 전송
            while self._cycle_idx < len(self._cycle_steps):
                step = self._cycle_steps[self._cycle_idx]
                kind = step[0]

                if kind == "status":
                    self.main.statusBar().showMessage(step[1])
                    self._cycle_idx += 1

                elif kind == "move":
                    _, mtype, p = step
                    fn = robot.add_move_lin if mtype == "lin" else robot.add_move_ptp
                    slot = fn(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])
                    if slot is None:
                        return  # KRL 큐 가득 — 이미 보낸 것 완료 후 재시도
                    self._cycle_pending_slots.append(slot)
                    self._cycle_idx += 1
                    # 계속 루프: 다음도 move/status 면 이어서 배치 전송

                elif kind == "move_axis":
                    # 관절 PTP — 티칭된 관절값으로 정확히 복귀 (해가 하나뿐이라
                    # A6 감김 방향이 결정적 = 호스 꼬임 리셋). Home/놓기 스텝 전용.
                    _, ax = step
                    cur_ax = robot.get_axis_position()  # 검증용: 감김 풀림을 숫자로 기록
                    if cur_ax:
                        logger.info(f"사이클 관절 이동: A6 {cur_ax['a6']:.1f}° → {ax['a6']:.1f}° (Δ {ax['a6'] - cur_ax['a6']:+.1f}°)")
                    slot = robot.add_move_axis(ax["a1"], ax["a2"], ax["a3"], ax["a4"], ax["a5"], ax["a6"])
                    if slot is None:
                        return  # KRL 큐 가득 — 재시도
                    self._cycle_pending_slots.append(slot)
                    self._cycle_idx += 1

                elif kind in ("vacuum", "blow", "dwell"):
                    # ★ 그리퍼/대기 스텝은 모든 모션의 물리적 완료가 선행 조건
                    if self._cycle_pending_slots:
                        return  # 다음 틱에서 1)이 완료 확인 후 여기로 돌아옴
                    if kind == "vacuum":
                        if not robot.set_vacuum(step[1]):
                            self._cycle_abort("진공 적용 실패 — ext_move 가 실행 중(#P_ACTIVE)인지 확인")
                            return
                        self._cycle_idx += 1
                    elif kind == "blow":
                        robot.vacuum_blow(step[1])
                        self._cycle_idx += 1
                    else:  # dwell
                        self._cycle_dwell_until = time.time() + float(step[1])
                        self._cycle_idx += 1
                        return  # 대기 시작

                else:
                    self._cycle_abort(f"알 수 없는 스텝: {kind}")
                    return

            # 4) 모든 스텝 소진 — 남은 모션 완료까지 기다린 뒤 종료
            if self._cycle_pending_slots:
                return
            self._cycle_timer.stop()
            self._cycle_active = False
            self.main.statusBar().showMessage(f"✅ {self._cycle_done_msg}")
            logger.info(self._cycle_done_msg)
            # 완료 콜백 (연속 픽 루프의 다음 사이클). 콜백이 새 사이클을 시작할 수
            # 있으므로 _cycle_active 를 먼저 False 로 내린 뒤 호출한다.
            cb = getattr(self, "_cycle_on_done", None)
            self._cycle_on_done = None
            self._cycle_on_abort = None
            if cb is not None:
                cb()
        except Exception as e:
            self._cycle_abort(f"사이클 오류: {e}")

    def _build_pick_steps(
        self,
        target: Dict[str, float],
        approach_dist: float,
        dwell: float,
        place: Optional[Dict[str, float]],
        home: Optional[Dict[str, float]],
        label: str = "",
    ) -> List[Tuple]:
        """상용 표준 픽 사이클 스텝 생성.

        Approach(PTP, 빠르게) → 하강(LIN, 정밀) → 진공 ON + 흡착대기 →
        상승(LIN, 잡은 채) → 놓기 위치(PTP) → 진공 OFF + 블로우 → (Home 복귀)
        """
        ax, ay, az = self._compute_approach_position(target, approach_dist)
        app_pose = dict(target)
        app_pose.update(x=ax, y=ay, z=az)
        pre = f"{label} " if label else ""

        # 놓기/Home 은 관절 티칭값이 있으면 관절 PTP 로 (호스 꼬임 방지 — _move_to_taught 와 같은 이유).
        # 호출자가 넘기는 place/home 은 항상 main 의 티칭 포즈이므로 대응 관절값도 main 에서 가져온다.
        place_axis = getattr(self.main, "place_axis", None)
        home_axis = getattr(self.main, "home_axis", None)

        steps: List[Tuple] = [
            ("status", f"{pre}① Approach 이동 (PTP)..."),
            ("move", "ptp", app_pose),
            ("status", f"{pre}② 하강 (LIN)..."),
            ("move", "lin", dict(target)),
            ("status", f"{pre}③ 진공 ON — 흡착 {dwell:.1f}s 대기..."),
            ("vacuum", True),
            ("dwell", dwell),
            ("status", f"{pre}④ 상승 (LIN, 잡은 채)..."),
            ("move", "lin", app_pose),
        ]
        if place is not None:
            if place_axis is not None:
                steps += [
                    ("status", f"{pre}⑤ 놓기 위치로 이동 (관절 PTP)..."),
                    ("move_axis", dict(place_axis)),
                ]
            else:
                steps += [
                    ("status", f"{pre}⑤ 놓기 위치로 이동 (PTP)..."),
                    ("move", "ptp", dict(place)),
                ]
            steps += [
                ("status", f"{pre}⑥ 놓기 — 진공 OFF + 블로우..."),
                ("vacuum", False),
                ("blow", 0.4),
            ]
        if home is not None:
            if home_axis is not None:
                steps += [
                    ("status", f"{pre}Home 복귀 (관절 PTP)..."),
                    ("move_axis", dict(home_axis)),
                ]
            else:
                steps += [
                    ("status", f"{pre}Home 복귀..."),
                    ("move", "ptp", dict(home)),
                ]
        return steps

    def _execute_pick_cycle(self):
        """선택된 대상 1개를 자동 픽 (잡기→놓기→Home). 상용 스타일 원버튼 사이클."""
        noun = self.SEQ_OBJECT_NOUN
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        if self.target_pose is None:
            QMessageBox.warning(self, "오류", f"먼저 {noun}를 선택하세요")
            return
        place = getattr(self.main, "place_pose", None)
        if place is None:
            QMessageBox.warning(
                self,
                "놓기 위치 없음",
                "놓기(Place) 위치가 저장되지 않았습니다.\n" "로봇을 놓을 자리로 조그한 뒤 '📍 놓기 위치 저장'을 먼저 누르세요.",
            )
            return

        target = dict(self.target_pose)
        offset = self.approach_dist.value()
        dwell = self.vac_dwell_spin.value()
        home = self.main.home_pose

        # Z 안전 검증: target + approach + place (+home)
        ax, ay, az = self._compute_approach_position(target, offset)
        for z in (target["z"], az, place["z"]) + ((home["z"],) if home else ()):
            if not self._validate_z(z):
                return

        speed = self._effective_speed(self.speed_spin.value())
        msg = [
            f"🤖 픽 사이클 실행 — {noun} 1개\n",
            f"속도: {speed}%" + (" (AUT 50% 상한)" if self._is_aut_mode() else ""),
            f"Approach 거리: {offset:.0f}mm, 흡착 대기: {dwell:.1f}s",
            "",
            "순서: Approach → 하강 → 진공ON → 상승 → 놓기 위치 → 진공OFF+블로우" + (" → Home" if home else ""),
            "",
            "⚠ ext_move 가 실행 중이어야 합니다 (진공 + 모션 모두)",
            "⚠ 비상시 Space 또는 비상정지 버튼",
            "\n진행하시겠습니까?",
        ]
        if QMessageBox.question(self, "픽 사이클 확인", "\n".join(msg), QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"속도 설정 실패:\n{e}")
            return
        steps = self._build_pick_steps(target, offset, dwell, place, home)
        self._run_cycle(steps, f"픽 사이클 완료 ({noun})")

    def _enqueue_pick(self):
        """선택된 대상의 픽(잡기→놓기) 액션을 시퀀스 큐에 추가."""
        noun = self.SEQ_OBJECT_NOUN
        if self.target_pose is None:
            QMessageBox.warning(self, "오류", f"먼저 {noun}를 선택하세요")
            return
        if not self._validate_z(self.target_pose["z"]):
            return
        idx_str = str(self.selected_idx + 1) if self.selected_idx is not None else "?"
        action = {
            "type": "object_pick",
            "label": f"🧲 {noun} #{idx_str} 픽 (잡기→놓기)",
            "target": dict(self.target_pose),
            "approach_dist": self.approach_dist.value(),
            "dwell": self.vac_dwell_spin.value(),
        }
        self.user_queue.append(action)
        self._refresh_action_list()
        self.main.statusBar().showMessage(f"➕ 시퀀스에 추가: {action['label']}")

    # ============================================================
    # 속도 제어
    # ============================================================

    def _on_speed_changed(self, value: int):
        """SpinBox 값 변경 시 - 자동 적용 안 함 (사용자가 '적용' 버튼 누를 때만)."""
        pass

    def _apply_speed_now(self):
        """현재 SpinBox 속도를 즉시 로봇에 적용 ($OV_PRO 변경)."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        speed = self.speed_spin.value()
        try:
            ok = self.main.robot.set_speed(speed)
            if ok:
                self.main.statusBar().showMessage(f"⚙ 속도 적용: {speed}% ($OV_PRO)")
                logger.info(f"속도 즉시 적용: {speed}%")
            else:
                self.main.statusBar().showMessage("속도 적용 실패")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"속도 적용 실패:\n{e}")

    # ============================================================
    # Home 관련
    # ============================================================

    def _set_home_to_current(self):
        """현재 로봇 TCP 위치를 새 Home으로 저장."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        cur = self.main.robot.get_tcp_position()
        cur_axis = self.main.robot.get_axis_position()  # 관절값도 티칭 — Home 복귀를 관절 PTP 로 (A6 감김 리셋)
        if cur is None:
            QMessageBox.critical(self, "오류", "현재 TCP 위치를 읽지 못했습니다")
            return

        old = self.main.home_pose
        old_str = (
            f"  X: {old['x']:.2f}, Y: {old['y']:.2f}, Z: {old['z']:.2f}\n" f"  A: {old['a']:.2f}, B: {old['b']:.2f}, C: {old['c']:.2f}"
            if old
            else "(저장된 Home 없음)"
        )
        new_str = f"  X: {cur['x']:.2f}, Y: {cur['y']:.2f}, Z: {cur['z']:.2f}\n" f"  A: {cur['a']:.2f}, B: {cur['b']:.2f}, C: {cur['c']:.2f}"
        msg = f"📍 Home 위치를 현재 위치로 재설정하시겠습니까?\n\n[기존 Home]\n{old_str}\n\n[새 Home (현재 TCP)]\n{new_str}"
        ret = QMessageBox.question(self, "Home 재설정 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        self.main.home_pose = cur
        self.main.home_axis = dict(cur_axis) if cur_axis else None
        if cur_axis is None:
            logger.warning("관절값($AXIS_ACT)을 읽지 못해 Home 은 Cartesian 으로만 저장됨 (관절 복귀 비활성)")
        self.main._save_teach_poses()  # 앱을 껐다 켜도 유지되도록 teach_poses.json 에 기록
        self.btn_move_home.setEnabled(True)
        self.btn_add_home_to_seq.setEnabled(True)
        self.main.statusBar().showMessage(f"📍 Home 재설정됨: X={cur['x']:.1f}, Y={cur['y']:.1f}, Z={cur['z']:.1f}")
        logger.info(f"Home 재설정: {cur}")

    @staticmethod
    def _axis_str(axis: Dict[str, float]) -> str:
        return ", ".join(f"A{i} {axis[f'a{i}']:.1f}°" for i in range(1, 7))

    def _move_to_taught(self, title: str, icon: str, pose: Dict[str, float], axis: Optional[Dict[str, float]]):
        """티칭 위치(Home/놓기)로 이동 — 관절값이 있으면 관절 PTP, 없으면 Cartesian PTP.

        관절 PTP 를 우선하는 이유 (진공 호스 꼬임 방지): Cartesian 목표는 같은 자세를
        만드는 관절 해가 여러 개라 로봇이 A6 를 감는 쪽을 고를 수 있지만, 관절 목표는
        해가 하나뿐이라 티칭 당시의 감김 상태로 정확히 복귀한다 = 누적이 매번 리셋된다.
        Z 안전 검증은 티칭된 Cartesian z 로 동일하게 수행한다 (관절 목표여도 도착점의
        높이는 티칭 때의 z 그대로이므로).
        """
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        if not self._validate_z(pose["z"]):
            return
        speed = self._effective_speed(self.speed_spin.value())

        if axis is not None:
            mode_line = "PTP (관절 좌표 복귀 — A6 감김 리셋, 호스 꼬임 방지)"
            target_str = "  " + self._axis_str(axis) + f"\n  (티칭 시 TCP: X {pose['x']:.1f}, Y {pose['y']:.1f}, Z {pose['z']:.1f})"
        else:
            mode_line = "PTP (Cartesian — 관절 티칭값 없음, 재티칭하면 관절 복귀 활성화)"
            target_str = (
                f"  X: {pose['x']:.2f}\n  Y: {pose['y']:.2f}\n  Z: {pose['z']:.2f}\n"
                f"  A: {pose['a']:.2f}\n  B: {pose['b']:.2f}\n  C: {pose['c']:.2f}"
            )
        msg = (
            f"{icon} {title}로 이동\n\n"
            f"방식: {mode_line}\n"
            f"속도: {speed}%" + (" (AUT 50% 상한 적용)" if self._is_aut_mode() else "") + "\n\n"
            f"목표:\n{target_str}\n\n"
            f"진행하시겠습니까?"
        )
        ret = QMessageBox.question(self, f"{title} 이동 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
            if axis is not None:
                # 검증용 로그: 현재 A6 → 목표 A6 (감김이 풀리는 방향/양을 숫자로 확인)
                cur_ax = self.main.robot.get_axis_position()
                if cur_ax:
                    logger.info(f"{title} 관절 복귀: A6 {cur_ax['a6']:.1f}° → {axis['a6']:.1f}° (Δ {axis['a6'] - cur_ax['a6']:+.1f}°)")
                slot = self.main.robot.add_move_axis(axis["a1"], axis["a2"], axis["a3"], axis["a4"], axis["a5"], axis["a6"])
            else:
                slot = self.main.robot.add_move_ptp(pose["x"], pose["y"], pose["z"], pose["a"], pose["b"], pose["c"])
            if slot is None:
                QMessageBox.critical(self, "오류", f"{title} 이동 명령 큐에 추가 실패")
                return
            kind = "관절" if axis is not None else "Cartesian"
            self.main.statusBar().showMessage(f"{icon} {title} 이동 명령 큐에 추가됨 (slot={slot}, {kind} PTP)")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"{title} 이동 실패:\n{e}")

    def _move_to_home(self):
        """저장된 Home 위치로 이동 — 관절 티칭값이 있으면 관절 PTP (_move_to_taught 참고)."""
        if self.main.home_pose is None:
            QMessageBox.warning(self, "오류", "Home 위치가 저장되지 않았습니다")
            return
        self._move_to_taught("Home", "🏠", self.main.home_pose, getattr(self.main, "home_axis", None))

    def _move_to_place(self):
        """저장된 놓기(Place) 위치로 이동 — 관절 티칭값이 있으면 관절 PTP (_move_to_taught 참고)."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        place = getattr(self.main, "place_pose", None)
        if place is None:
            QMessageBox.warning(
                self,
                "오류",
                "놓기(Place) 위치가 저장되지 않았습니다.\n로봇을 놓을 자리로 조그한 뒤 '놓기 위치 재설정'을 먼저 누르세요.",
            )
            return

        self._move_to_taught("놓기 위치", "📦", place, getattr(self.main, "place_axis", None))

    def _clear_motion_queue(self):
        """KRL 큐의 모든 슬롯을 0으로 리셋 (대기 중인 이동 취소)."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        try:
            ok = self.main.robot.clear_queue()
            if ok:
                self.main.statusBar().showMessage("🗑 큐 비움 - 대기 중이던 모든 이동 명령 취소")
                logger.info("모션 큐 비움")
            else:
                self.main.statusBar().showMessage("큐 비우기 부분 실패")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"큐 비우기 실패:\n{e}")

    # ============================================================
    # Approach 계산
    # ============================================================

    @staticmethod
    def _compute_approach_position(target: Dict[str, float], offset_mm: float):
        """target 자세의 Tool +Z 반대 방향으로 offset mm 떨어진 위치 (Approach/Retract 공통)."""
        T = tcp_to_homogeneous(target)
        z_axis = T[:3, 2]
        target_pos = T[:3, 3]
        approach_pos = target_pos - z_axis * offset_mm
        return float(approach_pos[0]), float(approach_pos[1]), float(approach_pos[2])

    # ============================================================
    # 시퀀스 큐 (Python 측 user_queue)
    # ============================================================

    def _refresh_action_list(self):
        self.action_list.clear()
        for i, action in enumerate(self.user_queue):
            self.action_list.addItem(f"{i + 1:2d}. {action['label']}")

    def _enqueue_object_move(self):
        """현재 선택된 대상 이동을 시퀀스 큐에 추가."""
        noun = self.SEQ_OBJECT_NOUN
        if self.target_pose is None:
            QMessageBox.warning(self, "오류", f"먼저 {noun}를 선택하세요")
            return
        if not self._validate_z(self.target_pose["z"]):
            return

        is_lin = self.move_mode_combo.currentText().startswith("LIN")
        idx_str = str(self.selected_idx + 1) if self.selected_idx is not None else "?"
        action = {
            "type": "object_move",
            "label": f"{noun} #{idx_str} 이동 [{'LIN' if is_lin else 'PTP'}, " f"{'A/T/R' if self.use_approach.isChecked() else 'T'}]",
            "target": dict(self.target_pose),
            "is_lin": is_lin,
            "use_approach": self.use_approach.isChecked(),
            "approach_dist": self.approach_dist.value(),
        }
        self.user_queue.append(action)
        self._refresh_action_list()
        self.main.statusBar().showMessage(f"➕ 시퀀스에 추가: {action['label']}")

    def _enqueue_home_to_sequence(self):
        if self.main.home_pose is None:
            QMessageBox.warning(self, "오류", "Home 위치가 저장되지 않았습니다")
            return
        home_axis = getattr(self.main, "home_axis", None)
        action = {
            "type": "home",
            "label": "🏠 Home 이동",
            "target": dict(self.main.home_pose),
            # 추가 시점의 관절값 스냅샷 (target 과 같은 원칙) — 있으면 관절 PTP 로 실행
            "axis": dict(home_axis) if home_axis else None,
        }
        self.user_queue.append(action)
        self._refresh_action_list()
        self.main.statusBar().showMessage("➕ 시퀀스에 추가: Home 이동")

    def _remove_selected_action(self):
        row = self.action_list.currentRow()
        if 0 <= row < len(self.user_queue):
            removed = self.user_queue.pop(row)
            self._refresh_action_list()
            self.main.statusBar().showMessage(f"❌ 시퀀스 항목 제거: {removed['label']}")

    def _clear_user_queue(self):
        if not self.user_queue:
            return
        ret = QMessageBox.question(
            self,
            "확인",
            f"시퀀스 큐의 {len(self.user_queue)}개 항목을 모두 제거하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret == QMessageBox.Yes:
            self.user_queue.clear()
            self._refresh_action_list()
            self.main.statusBar().showMessage("시퀀스 큐 비움")

    def _start_sequence(self):
        """시퀀스 큐의 모든 액션을 KRL 큐에 순차 전송."""
        if not self.user_queue:
            QMessageBox.information(self, "알림", "시퀀스 큐가 비어 있습니다")
            return
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return

        speed = self._effective_speed(self.speed_spin.value())

        has_pick = any(a["type"] == "object_pick" for a in self.user_queue)
        place = getattr(self.main, "place_pose", None)
        if has_pick and place is None:
            QMessageBox.warning(
                self,
                "놓기 위치 없음",
                "시퀀스에 픽 액션이 있는데 놓기(Place) 위치가 저장되지 않았습니다.\n" "'📍 놓기 위치 저장'을 먼저 누르세요.",
            )
            return
        if has_pick and place is not None and not self._validate_z(place["z"]):
            return

        for action in self.user_queue:
            if not self._validate_z(action["target"]["z"]):
                return
            # approach 지점 Z도 검증
            # (Tool +Z가 옆/위를 향하면 approach가 바닥 한계 아래로 갈 수 있음)
            if action["type"] in ("object_move", "object_pick") and action.get("use_approach", True):
                ax, ay, az = self._compute_approach_position(action["target"], action.get("approach_dist", 50))
                if not self._validate_z(az):
                    return

        msg_lines = [
            f"▶ 시퀀스 실행 ({len(self.user_queue)}개 액션)\n",
            f"속도: {speed}%" + (" (AUT 모드 → 50% 상한 적용)" if self._is_aut_mode() else ""),
            f"Z 한계: {self.z_min_spin.value()}mm 이상",
            "",
            "실행 순서:",
        ]
        for i, a in enumerate(self.user_queue):
            msg_lines.append(f"  {i + 1}. {a['label']}")
        msg_lines.append("")
        if self._is_aut_mode():
            msg_lines.append("⚠ AUT 모드 — 시작 즉시 자동 이동")
        else:
            msg_lines.append("⚠ T1 모드 — SmartPAD 데드맨+시작 버튼 필요")
        msg_lines.append("⚠ 비상시 Space 또는 비상정지 버튼")
        msg_lines.append("\n진행하시겠습니까?")

        ret = QMessageBox.question(
            self,
            "시퀀스 실행 확인",
            "\n".join(msg_lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"속도 설정 실패:\n{e}")
            return

        # 액션 → 스텝 전개 후 단계별 실행기로 실행.
        # (모든 슬롯을 한꺼번에 KRL 로 던지는 이전 방식으로는 픽 중간의
        #  진공 ON/OFF 를 끼워넣을 수 없어 단계별 실행으로 통일)
        all_steps: List[Tuple] = []
        n = len(self.user_queue)
        for i, action in enumerate(self.user_queue):
            prefix = f"[{i + 1}/{n}]"
            all_steps.append(("status", f"{prefix} {action['label']}"))
            all_steps.extend(self._expand_action_to_steps(action, prefix))
        self._run_cycle(all_steps, f"시퀀스 완료 ({n}개 액션)")

    def _expand_action_to_steps(self, action: Dict, prefix: str = "") -> List[Tuple]:
        """user_queue 액션 1개 → 실행기 스텝 목록."""
        target = action["target"]
        a_type = action["type"]

        if a_type == "home":
            axis = action.get("axis")
            if axis:
                return [("move_axis", dict(axis))]  # 관절 티칭값 있음 → A6 감김 리셋 복귀
            return [("move", "ptp", dict(target))]  # 구버전 액션/관절값 없음 → Cartesian PTP

        if a_type == "object_move":
            is_lin = action.get("is_lin", True)
            kind = "lin" if is_lin else "ptp"
            if action.get("use_approach", True):
                ax, ay, az = self._compute_approach_position(target, action.get("approach_dist", 50))
                app_pose = dict(target)
                app_pose.update(x=ax, y=ay, z=az)
                return [
                    ("move", kind, app_pose),
                    ("move", "lin", dict(target)),
                    ("move", "lin", app_pose),
                ]
            return [("move", kind, dict(target))]

        if a_type == "object_pick":
            # 시퀀스 픽은 Home 복귀를 포함하지 않음 (사용자가 'Home 추가'로 명시)
            return self._build_pick_steps(
                dict(target),
                action.get("approach_dist", 50),
                action.get("dwell", 0.5),
                getattr(self.main, "place_pose", None),
                home=None,
                label=prefix,
            )

        logger.warning(f"알 수 없는 액션 타입: {a_type}")
        return []
