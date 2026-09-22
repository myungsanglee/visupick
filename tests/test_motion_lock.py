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

    # 모든 모달 대화상자를 막는다. 특히 information 은 '영상 저장 완료' 안내로 뜨는데,
    # 패치하지 않으면 테스트가 창을 띄운 채 영원히 멈춘다.
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
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


class TestLiveRespectsRoi:
    """실시간 미리보기도 Bin Box(ROI) 밖 검출은 정식 경로와 똑같이 걸러야 한다.

    데모 영상에 작업 영역 밖 물체가 잡히면 안 되기 때문. 포인트클라우드를 안 뽑으므로
    깊이 밴드(2차 게이트)는 생략되고 2D 사각형 게이트만 적용된다.
    """

    @pytest.fixture
    def two_objects(self, tab, monkeypatch):
        """ROI 안 1개 + 밖 1개를 내놓는 검출기로 교체."""
        inside = {"bbox": [40.0, 30.0, 120.0, 90.0], "confidence": 0.9, "class_id": 0, "class_name": "in", "mask": None}
        outside = {"bbox": [130.0, 95.0, 158.0, 118.0], "confidence": 0.8, "class_id": 0, "class_name": "out", "mask": None}

        class Sam3Two:
            loaded = True

            def detect(self, img, conf, prompt=""):
                return [dict(inside), dict(outside)], 100.0

        monkeypatch.setattr(tab, "_sam3", Sam3Two())
        return tab

    def test_outside_roi_is_dropped(self, two_objects, monkeypatch):
        t = two_objects
        monkeypatch.setattr(t, "roi_2d", (30.0, 20.0, 125.0, 95.0))
        monkeypatch.setattr(t, "roi_3d", None)
        t._toggle_live_sam3()
        try:
            t._live_tick()
            labels = [b[5] for b in t.view_2d._overlay_boxes]  # (x1,y1,x2,y2,color,label,idx)
            assert len(labels) == 1, f"ROI 밖 검출이 남음: {labels}"
            assert labels[0].startswith("in")
        finally:
            t._toggle_live_sam3()

    def test_no_roi_shows_everything(self, two_objects, monkeypatch):
        """ROI 가 없으면(빈 박스 미설정) 정식 경로와 마찬가지로 전부 보여준다."""
        t = two_objects
        monkeypatch.setattr(t, "roi_2d", None)
        t._toggle_live_sam3()
        try:
            t._live_tick()
            assert len(t.view_2d._overlay_boxes) == 2
        finally:
            t._toggle_live_sam3()

    def test_partial_overlap_excluded(self, two_objects, monkeypatch):
        """bbox 가 ROI 경계에 걸치면 제외 — 정식 경로와 같은 규칙(전체 포함만 통과)."""
        t = two_objects
        monkeypatch.setattr(t, "roi_2d", (30.0, 20.0, 100.0, 95.0))  # 안쪽 객체의 우측이 삐져나감
        monkeypatch.setattr(t, "roi_3d", None)
        t._toggle_live_sam3()
        try:
            t._live_tick()
            assert len(t.view_2d._overlay_boxes) == 0
        finally:
            t._toggle_live_sam3()


class TestRoiFilterShared:
    """정식 경로와 실시간이 같은 필터를 쓰는지 (규칙이 갈라지지 않게)."""

    def test_same_helper_used(self, tab, monkeypatch):
        monkeypatch.setattr(tab, "roi_2d", (0.0, 0.0, 100.0, 100.0))
        monkeypatch.setattr(tab, "roi_3d", None)
        dets = [
            {"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0, "class_name": "in", "mask": None},
            {"bbox": [90.0, 90.0, 140.0, 140.0], "confidence": 0.9, "class_id": 0, "class_name": "out", "mask": None},
        ]
        kept = tab._filter_by_roi(dets)
        assert [d["class_name"] for d in kept] == ["in"]

    def test_depth_gate_skipped_without_xyz(self, tab, monkeypatch):
        """xyz 를 안 넘기면 깊이 밴드는 건너뛴다 (실시간 경로)."""
        monkeypatch.setattr(tab, "roi_2d", (0.0, 0.0, 100.0, 100.0))
        monkeypatch.setattr(tab, "roi_3d", {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1})
        dets = [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0, "class_name": "in", "mask": None}]
        assert len(tab._filter_by_roi(dets)) == 1  # 깊이로 걸리지 않음


class TestLiveRecording:
    """실시간 영상 저장 (mp4) — 데모 촬영용.

    '실시간 영상 저장' 버튼은 실시간 검출 중에만 보이고, 누르면 그 시점부터 녹화한다.
    '실시간 중지' 를 누르면 파일이 완성된다.
    """

    def test_button_hidden_until_live(self, tab):
        assert not tab.btn_save_live.isVisibleTo(tab)
        tab._toggle_live_sam3()
        try:
            assert tab.btn_save_live.isVisibleTo(tab)
        finally:
            tab._toggle_live_sam3()
        assert not tab.btn_save_live.isVisibleTo(tab)

    def test_records_playable_mp4(self, tab, tmp_path, monkeypatch):
        import time

        import cv2

        out = tmp_path / "demo.mp4"
        monkeypatch.setattr(tab.main, "ask_debug_filename", lambda *a, **k: out)

        tab._toggle_live_sam3()
        try:
            tab._live_tick()  # 녹화 전 프레임
            tab._start_live_recording()
            assert tab._rec_writer is not None
            t0 = time.time()
            while time.time() - t0 < 0.5:
                tab._live_tick()
                time.sleep(0.02)
        finally:
            tab._toggle_live_sam3()  # 중지 → 저장

        assert tab._rec_writer is None, "중지 후에도 writer 가 남아 있음"
        assert out.exists() and out.stat().st_size > 0

        cap = cv2.VideoCapture(str(out))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ok, frame = cap.read()
        finally:
            cap.release()
        assert ok and frame is not None, "mp4 디코드 실패"
        assert n > 0 and abs(fps - tab.REC_FPS) < 1

    def test_playback_duration_matches_wall_clock(self, tab, tmp_path, monkeypatch):
        """검출이 느려도 재생 속도가 실시간과 맞아야 한다 (부족한 만큼 프레임 복제)."""
        import time

        out = tmp_path / "speed.mp4"
        monkeypatch.setattr(tab.main, "ask_debug_filename", lambda *a, **k: out)
        tab._toggle_live_sam3()
        try:
            tab._start_live_recording()
            t0 = time.time()
            for _ in range(3):  # 프레임은 3장뿐이지만 실제로는 0.6초가 흐른다
                tab._live_tick()
                time.sleep(0.2)
            elapsed = time.time() - t0
        finally:
            tab._toggle_live_sam3()
        # 출력 프레임 수 / REC_FPS ≈ 실제 경과 시간
        assert tab._live_frames == 3
        # (중지 시점에 _rec_written 이 0 으로 리셋되므로 파일에서 길이를 읽는다)
        import cv2

        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert abs(n / tab.REC_FPS - elapsed) < 0.35, f"재생 {n / tab.REC_FPS:.2f}초 vs 실제 {elapsed:.2f}초"

    def test_stop_without_recording_is_harmless(self, tab):
        tab._toggle_live_sam3()
        tab._toggle_live_sam3()  # 녹화 안 켜고 중지
        assert tab._rec_writer is None

    def test_double_start_does_not_replace_file(self, tab, tmp_path, monkeypatch):
        out = tmp_path / "once.mp4"
        monkeypatch.setattr(tab.main, "ask_debug_filename", lambda *a, **k: out)
        tab._toggle_live_sam3()
        try:
            tab._start_live_recording()
            first = tab._rec_path
            tab._start_live_recording()  # 이미 녹화 중 — 무시돼야 한다
            assert tab._rec_path is first
        finally:
            tab._toggle_live_sam3()
