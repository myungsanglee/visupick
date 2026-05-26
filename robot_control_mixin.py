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

import logging
from typing import Optional, List, Dict

import numpy as np  # noqa: F401  (탭 코드와의 일관성 위해 유지)

from PySide6.QtWidgets import QMessageBox

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
        """로봇 연결 시 main이 호출. Home 관련 버튼 활성화."""
        self.btn_set_home.setEnabled(True)
        if self.main.home_pose:
            self.btn_move_home.setEnabled(True)
            self.btn_add_home_to_seq.setEnabled(True)
        self._refresh_mode_display()

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
            f"  X: {old['x']:.2f}, Y: {old['y']:.2f}, Z: {old['z']:.2f}\n"
            f"  A: {old['a']:.2f}, B: {old['b']:.2f}, C: {old['c']:.2f}"
            if old
            else "(저장된 Home 없음)"
        )
        new_str = (
            f"  X: {cur['x']:.2f}, Y: {cur['y']:.2f}, Z: {cur['z']:.2f}\n"
            f"  A: {cur['a']:.2f}, B: {cur['b']:.2f}, C: {cur['c']:.2f}"
        )
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
        """저장된 Home 위치로 LIN 이동."""
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
            f"방식: LIN (직선)\n"
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
            slot = self.main.robot.add_move_lin(h["x"], h["y"], h["z"], h["a"], h["b"], h["c"])
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
            "label": f"{noun} #{idx_str} 이동 [{'LIN' if is_lin else 'PTP'}, "
            f"{'A/T/R' if self.use_approach.isChecked() else 'T'}]",
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

        for action in self.user_queue:
            if not self._validate_z(action["target"]["z"]):
                return
            # object_move + approach인 경우 계산된 approach 지점 Z도 검증
            # (Tool +Z가 옆/위를 향하면 approach가 바닥 한계 아래로 갈 수 있음)
            if action["type"] == "object_move" and action.get("use_approach", True):
                ax, ay, az = self._compute_approach_position(
                    action["target"], action.get("approach_dist", 50)
                )
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
            total_slots = []
            for action in self.user_queue:
                slots = self._send_action_to_krl_queue(action)
                if slots is None:
                    QMessageBox.critical(self, "오류", f"시퀀스 중단: '{action['label']}' 추가 실패")
                    return
                total_slots.extend(slots)
            self.main.statusBar().showMessage(f"✅ 시퀀스 시작: {len(self.user_queue)}개 액션 → {len(total_slots)}개 KRL 슬롯")
            logger.info(f"시퀀스 실행: actions={len(self.user_queue)}, krl_slots={total_slots}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"시퀀스 실행 실패:\n{e}")

    def _send_action_to_krl_queue(self, action: Dict) -> Optional[List[int]]:
        """단일 액션을 KRL 큐에 추가 (필요 시 여러 슬롯)."""
        slots = []
        target = action["target"]
        a_type = action["type"]

        if a_type == "home":
            slot = self.main.robot.add_move_lin(
                target["x"], target["y"], target["z"], target["a"], target["b"], target["c"]
            )
            if slot is None:
                return None
            slots.append(slot)
            return slots

        if a_type == "object_move":
            is_lin = action.get("is_lin", True)
            use_approach = action.get("use_approach", True)
            offset = action.get("approach_dist", 50)

            add_motion = self.main.robot.add_move_lin if is_lin else self.main.robot.add_move_ptp

            if use_approach:
                ax, ay, az = self._compute_approach_position(target, offset)
                s1 = add_motion(ax, ay, az, target["a"], target["b"], target["c"])
                if s1 is None:
                    return None
                slots.append(s1)
                s2 = self.main.robot.add_move_lin(
                    target["x"], target["y"], target["z"], target["a"], target["b"], target["c"]
                )
                if s2 is None:
                    return None
                slots.append(s2)
                s3 = self.main.robot.add_move_lin(ax, ay, az, target["a"], target["b"], target["c"])
                if s3 is None:
                    return None
                slots.append(s3)
            else:
                s = add_motion(target["x"], target["y"], target["z"], target["a"], target["b"], target["c"])
                if s is None:
                    return None
                slots.append(s)
            return slots

        return None
