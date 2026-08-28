"""line_tracking.py 테스트 — 합성 선 이미지로 검출→추적→샘플링 파이프라인 검증."""

import cv2
import numpy as np
import pytest

from line_tracking import detect_line, trace_path_on_skeleton, sample_path_by_3d_distance


def line_image(p0=(10, 50), p1=(190, 50), thickness=3, color=(0, 0, 0)):
    img = np.full((100, 200, 3), 255, np.uint8)
    cv2.line(img, p0, p1, color, thickness)
    return img


def flat_xyz(h=100, w=200, mm_per_px=1.0):
    """평평한 XYZ 맵 — x 가 픽셀 좌표에 비례 (3D 거리 = 픽셀 거리)."""
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.arange(w)[None, :] * mm_per_px
    xyz[:, :, 1] = np.arange(h)[:, None] * mm_per_px
    return xyz


class TestDetectLine:
    def test_straight_black_line(self):
        skel, dbg = detect_line(line_image())
        assert skel is not None and skel.any()
        ys, xs = np.nonzero(skel)
        assert ys.std() < 2.0  # 수평선이므로 스켈레톤 y 가 거의 일정

    def test_blank_image_no_line(self):
        skel, dbg = detect_line(np.full((100, 200, 3), 255, np.uint8))
        assert skel is None or skel.sum() < 10

    def test_color_marker_mode(self):
        img = line_image(color=(0, 0, 220))  # BGR 빨강 선
        skel, dbg = detect_line(img, color_mode="red")
        assert skel is not None and skel.any()


class TestTracePath:
    def test_path_connects_endpoints(self):
        skel, _ = detect_line(line_image())
        path = trace_path_on_skeleton(skel, (10, 50), (190, 50))
        assert path is not None and len(path) > 150
        # 경로 양끝이 클릭 지점 근처여야 한다
        assert abs(path[0][0] - 10) < 8 and abs(path[-1][0] - 190) < 8

    def test_l_shaped_path(self):
        img = np.full((200, 200, 3), 255, np.uint8)
        cv2.line(img, (20, 20), (20, 180), (0, 0, 0), 3)
        cv2.line(img, (20, 180), (180, 180), (0, 0, 0), 3)
        skel, _ = detect_line(img)
        path = trace_path_on_skeleton(skel, (20, 20), (180, 180))
        assert path is not None
        # BFS 는 스켈레톤 위로만 가므로 경로 길이는 L 자 (≈160+160)
        assert len(path) > 280

    def test_disconnected_returns_none(self):
        skel = np.zeros((100, 200), np.uint8)
        skel[50, 10:60] = 1
        skel[50, 120:190] = 1  # 끊어진 두 조각
        path = trace_path_on_skeleton(skel, (12, 50), (188, 50))
        assert path is None


class TestSampling:
    def test_uniform_spacing(self):
        skel, _ = detect_line(line_image())
        path = trace_path_on_skeleton(skel, (10, 50), (190, 50))
        idx = sample_path_by_3d_distance(path, flat_xyz(), 30.0)
        assert idx[0] == 0
        pts = [path[i] for i in idx]
        # 연속 샘플 간 3D(=픽셀) 거리가 ~30mm
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            assert d == pytest.approx(30.0, abs=3.0)
