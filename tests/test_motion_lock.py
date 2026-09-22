"""SAM3 실시간 검출 중 로봇 이동 잠금 (안전 기능).

실시간 검출은 프레임마다 화면을 갈아엎으므로, 그 사이에 픽/이동이 돌면 "화면에 보이는
것"과 "로봇이 가는 좌표"가 어긋난다. 그래서 실행 중에는 모션을 막는다.

버튼 비활성화만으로는 부족하다 — 단축키·시퀀스·타이머 콜백처럼 버튼을 거치지 않는
경로가 있기 때문. 그래서 **모션 진입점마다 하드 가드**가 있고, 이 파일이 그 가드가
실제로 먹는지(그리고 비상정지는 막히지 않는지) 확인한다.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tab(qapp):
    """빈 픽킹 탭 + 스텁 로봇/카메라/검출기. 실제 하드웨어 없이 잠금 로직만 본다."""
    from PySide6.QtWidgets import QMessageBox

    import main as M

    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

    w = M.VisuPickApp()
    t = w.bin_picking_tab
    t._sent = []
    t._estops = []

    class Robot:
        def get_tcp_position(self):
            return {"x": 0, "y": 0, "z": 300, "a": 0, "b": 180, "c": 0}

        def get_axis_position(self):
            return {"a1": 0, "a2": -90, "a3": 90, "a4": 0, "a5": 90, "a6": -45}

        def set_speed(self, v):
            pass

        def add_move_ptp(self, *a):
            t._sent.append("ptp")
            return 1

        def add_move_lin(self, *a):
            t._sent.append("lin")
            return 1

        def add_move_axis(self, *a):
            t._sent.append("axis")
            return 1

        def read_variable(self, n):
            return "0"

        def clear_queue(self):
            return True

        def emergency_stop(self):
            t._estops.append(1)

        def emergency_stop_release(self):
            pass

        def disconnect(self):
            pass

    class Cam:
        connected = True
        is_capture_ready = True

        def capture(self):
            return "frame"

        def frame_to_2d_image(self, f):
            return np.zeros((120, 160, 3), np.uint8)

        def disconnect(self):
            pass

    class Sam3:
        loaded = True

        def detect(self, img, conf, prompt=""):
            m = np.zeros(img.shape[:2], bool)
            m[30:90, 40:120] = True
            return [{"bbox": [40.0, 30.0, 120.0, 90.0], "confidence": 0.9, "class_id": 0, "class_name": "case", "mask": m}], 100.0

    w.robot = Robot()
    w.camera = Cam()
    w.home_pose = {"x": 0, "y": 0, "z": 300, "a": 0, "b": 180, "c": 0}
    w.place_pose = dict(w.home_pose)
    w.home_axis = w.place_axis = None
    t._sam3 = Sam3()
    t._on_robot_connected()
    t.target_pose = dict(w.home_pose)
    t.sam3_prompt_input.setText("case")
    yield t
    t._live_stop("정리")
    w.robot = None
    w.camera = None


@pytest.fixture
def live(tab):
    """실시간 검출을 켠 상태로 테스트를 돌리고, 끝나면 반드시 끈다."""
    tab._toggle_live_sam3()
    assert tab._live_running
    tab._sent.clear()
    yield tab
    if tab._live_running:
        tab._toggle_live_sam3()


class TestMotionBlockedWhileLive:
    @pytest.mark.parametrize("entry", ["_move_to_home", "_move_to_place", "_execute_move", "_auto_start"])
    def test_entry_points_send_nothing(self, live, entry):
        """버튼을 거치지 않고 직접 호출해도 모션이 나가면 안 된다."""
        getattr(live, entry)()
        assert live._sent == [], f"{entry}: 모션이 전송됨 {live._sent}"

    def test_run_cycle_blocked(self, live):
        live._run_cycle([("move", "ptp", {"x": 0, "y": 0, "z": 300, "a": 0, "b": 180, "c": 0})], "테스트")
        assert live._sent == []
        assert not live._cycle_is_running()

    def test_motion_buttons_disabled(self, live):
        for name in ("btn_move_home", "btn_move_place", "btn_pick_cycle", "btn_auto_start", "btn_start_seq"):
            btn = getattr(live, name, None)
            if btn is not None:
                assert not btn.isEnabled(), f"{name} 이 잠기지 않음"

    def test_emergency_stop_still_works(self, live):
        """잠금은 '이동'만 막는다 — 멈추는 수단은 절대 막히면 안 된다."""
        assert live.btn_estop.isEnabled()
        assert live.btn_clear_queue.isEnabled()
        live._estops.clear()
        live._emergency_stop()
        assert live._estops, "비상정지가 전달되지 않음"

    def test_blocked_reason_is_explained(self, live):
        reason = live._motion_blocked_reason()
        assert reason and "실시간" in reason


class TestLockReleasedOnStop:
    def test_motion_allowed_again(self, tab):
        tab._toggle_live_sam3()
        tab._toggle_live_sam3()  # 시작했다가 중지
        assert not tab._live_running
        assert tab._motion_blocked_reason() is None
        tab._sent.clear()
        tab._move_to_home()
        assert tab._sent, "중지 후에도 이동이 막혀 있음"

    def test_button_states_restored_not_blindly_enabled(self, tab):
        """복구는 '원래 상태'로 — 로봇 미연결처럼 원래 꺼져 있어야 할 버튼은 꺼진 채여야."""
        tab.btn_pick_cycle.setEnabled(False)  # 원래 비활성인 상황을 만든다
        tab._toggle_live_sam3()
        tab._toggle_live_sam3()
        assert not tab.btn_pick_cycle.isEnabled(), "원래 꺼져 있던 버튼이 켜짐"
        tab.btn_pick_cycle.setEnabled(True)


class TestLivePreviewIsSideEffectFree:
    def test_does_not_touch_capture_state(self, live):
        """데모용이므로 정식 캡처 상태(current_image/detections/pick_objects)를 바꾸면 안 된다."""
        before = (live.current_image, list(live.detections), list(live.pick_objects))
        for _ in range(3):
            live._live_tick()
        assert live.current_image is before[0]
        assert live.detections == before[1]
        assert live.pick_objects == before[2]

    def test_draws_overlays(self, live):
        live._live_tick()
        assert len(live.view_2d._overlay_boxes) == 1
        assert len(live.view_2d._overlay_masks) == 1
