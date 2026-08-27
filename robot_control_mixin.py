"""
RobotControlMixin — BinPickingTab과 CADMatchingTab이 공유하는 로봇 제어 /
시퀀스 큐 / 안전 기능 메서드 모음.

두 탭에서 중복되던 19개 메서드를 이 믹스인으로 추출했다. 동작은 완전히 동일하며
(객체/인스턴스 명사 차이만 SEQ_OBJECT_NOUN 클래스 속성으로 매개변수화),
각 탭은 `class XxxTab(RobotControlMixin, QWidget)` 형태로 상속한다.

이 믹스인은 각 탭이 _init_ui에서 생성하는 다음 속성에 의존한다 (여기서 만들지 않음):
  self.main (.robot, .home_pose, .statusBar()), self.speed_spin, self.z_min_spin,
  self.move_mode_combo, self.use_approach, self.approach_dist, self.target_pose,
  self.selected_idx, self.user_queue, self.action_list, self.btn_move,
  self.btn_add_obj_to_seq, self.btn_move_home, self.btn_set_home,
  self.btn_add_home_to_seq, self._current_mode
또한 각 탭 고유의 self._refresh_mode_display()를 호출한다 (MRO상 탭 구현이 사용됨).
"""

import time
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np  # noqa: F401  (탭 코드와의 일관성 위해 유지)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QDoubleSpinBox

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
        for name in ("btn_vac_on", "btn_vac_off", "btn_vac_blow", "btn_pick_cycle", "btn_set_place", "btn_add_pick_to_seq", "btn_auto_start"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(True)
        self._refresh_mode_display()

    # ============================================================
    # 진공 그리퍼 (SMC ZK2 — $OUT[7]=VAC_ON / $OUT[8]=VAC_Blow)
    # ============================================================

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
        pick_row.addWidget(self.btn_pick_cycle)

        self.btn_set_place = QPushButton("📍 놓기 위치 저장")
        self.btn_set_place.setToolTip("현재 로봇 TCP 위치를 '놓기(Place) 위치'로 저장 — 픽 사이클이 여기에 내려놓음")
        self.btn_set_place.clicked.connect(self._set_place_to_current)
        self.btn_set_place.setEnabled(False)
        pick_row.addWidget(self.btn_set_place)

        pick_row.addWidget(QLabel("흡착대기(s)"))
        self.vac_dwell_spin = QDoubleSpinBox()
        self.vac_dwell_spin.setRange(0.1, 5.0)
        self.vac_dwell_spin.setSingleStep(0.1)
        self.vac_dwell_spin.setValue(0.5)
        self.vac_dwell_spin.setFixedWidth(70)
        self.vac_dwell_spin.setToolTip("진공 ON 후 상승 전 대기 시간 (흡착이 자리잡을 시간)")
        pick_row.addWidget(self.vac_dwell_spin)
        pick_row.addStretch()
        vbox.addLayout(pick_row)
        return vbox

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
    # 스텝 형식: ("status", 문구) / ("move", "ptp"|"lin", pose_dict)
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
            steps += [
                ("status", f"{pre}⑤ 놓기 위치로 이동 (PTP)..."),
                ("move", "ptp", dict(place)),
                ("status", f"{pre}⑥ 놓기 — 진공 OFF + 블로우..."),
                ("vacuum", False),
                ("blow", 0.4),
            ]
        if home is not None:
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
        self.btn_move_home.setEnabled(True)
        self.btn_add_home_to_seq.setEnabled(True)
        self.main.statusBar().showMessage(f"📍 Home 재설정됨: X={cur['x']:.1f}, Y={cur['y']:.1f}, Z={cur['z']:.1f}")
        logger.info(f"Home 재설정: {cur}")

    def _move_to_home(self):
        """저장된 Home 위치로 PTP 이동 (먼 거리 복귀라 관절 공간 이동이 빠르고 안전)."""
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return
        if self.main.home_pose is None:
            QMessageBox.warning(self, "오류", "Home 위치가 저장되지 않았습니다")
            return

        h = self.main.home_pose
        if not self._validate_z(h["z"]):
            return
        speed = self._effective_speed(self.speed_spin.value())
        msg = (
            f"🏠 Home 위치로 이동\n\n"
            f"방식: PTP (관절)\n"
            f"속도: {speed}%" + (" (AUT 50% 상한 적용)" if self._is_aut_mode() else "") + "\n\n"
            f"목표:\n"
            f"  X: {h['x']:.2f}\n  Y: {h['y']:.2f}\n  Z: {h['z']:.2f}\n"
            f"  A: {h['a']:.2f}\n  B: {h['b']:.2f}\n  C: {h['c']:.2f}\n\n"
            f"진행하시겠습니까?"
        )
        ret = QMessageBox.question(self, "Home 이동 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)
            slot = self.main.robot.add_move_ptp(h["x"], h["y"], h["z"], h["a"], h["b"], h["c"])
            if slot is None:
                QMessageBox.critical(self, "오류", "Home 이동 명령 큐에 추가 실패")
                return
            self.main.statusBar().showMessage(f"🏠 Home 이동 명령 큐에 추가됨 (slot={slot})")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"Home 이동 실패:\n{e}")

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
        action = {
            "type": "home",
            "label": "🏠 Home 이동",
            "target": dict(self.main.home_pose),
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
            return [("move", "ptp", dict(target))]  # Home 복귀는 PTP 로 통일

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
