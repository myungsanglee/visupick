"""image_view 의 중심 정보 라벨 테스트 (Qt 앱 불필요 — staticmethod 라 직접 호출).

검출된 객체 중심에 'X: .., Y: .., Deg: .., Open: ..' 라벨을 객체 색으로 채우고
흰 글씨로 그린다. 가장자리 객체에서 라벨이 화면 밖으로 나가지 않는지가 핵심.
"""

import numpy as np

from image_view import DraggableImageLabel as V


def blank(h=200, w=400):
    return np.zeros((h, w, 3), np.uint8)


class TestCenterLabel:
    def test_fills_with_object_color_and_white_text(self):
        c = blank()
        V._draw_center_label(c, "X: 100.0, Y: 100.0, Deg: 12.3, Open: 0.42", 200, 100, (255, 80, 80))
        px = c.reshape(-1, 3)
        colors = {tuple(v) for v in np.unique(px, axis=0)}
        assert (255, 80, 80) in colors, "객체 색 채움 없음"
        assert (255, 255, 255) in colors, "흰 글씨 없음"

    def test_centered_on_point(self):
        c = blank()
        cx, cy = 200, 100
        V._draw_center_label(c, "X: 1.0, Y: 1.0, Deg: 1.0", cx, cy, (0, 200, 0))
        ys, xs = np.nonzero(np.any(c > 0, axis=2))
        assert abs((xs.min() + xs.max()) // 2 - cx) <= 2
        assert abs((ys.min() + ys.max()) // 2 - cy) <= 2

    def test_clamped_inside_canvas(self):
        """가장자리 객체 — 라벨이 잘려 안 보이는 것보다 안으로 밀어 넣는 편이 낫다."""
        for cx, cy in [(5, 5), (395, 195), (0, 100), (200, 0)]:
            c = blank()
            V._draw_center_label(c, "X: 999.0, Y: 999.0, Deg: 180.0, Open: 1.00", cx, cy, (80, 80, 255))
            ys, xs = np.nonzero(np.any(c > 0, axis=2))
            assert xs.min() >= 0 and xs.max() < c.shape[1], f"({cx},{cy}) 가로 벗어남"
            assert ys.min() >= 0 and ys.max() < c.shape[0], f"({cx},{cy}) 세로 벗어남"

    def test_box_grows_with_text(self):
        """글자 길이에 맞춰 사각형이 커져야 한다 (Open 유무로 길이가 달라짐)."""

        def width(text):
            c = blank()
            V._draw_center_label(c, text, 200, 100, (200, 200, 200))
            xs = np.nonzero(np.any(c > 0, axis=2))[1]
            return xs.max() - xs.min()

        assert width("X: 1.0, Y: 1.0, Deg: 1.0, Open: 0.50") > width("X: 1.0, Y: 1.0, Deg: 1.0")

    def test_numpy_color_accepted(self):
        """팔레트 색이 numpy 정수로 올 수도 있다 — cv2 가 거부하지 않게 int 변환."""
        c = blank()
        V._draw_center_label(c, "X: 1.0", 200, 100, np.array([10, 20, 30], dtype=np.int64))
        assert c.any()
