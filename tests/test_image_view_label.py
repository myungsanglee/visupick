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
        V._draw_info_label(c, "X: 100.0, Y: 100.0, Deg: 12.3, Open: 0.42", 200, 100, (255, 80, 80))
        px = c.reshape(-1, 3)
        colors = {tuple(v) for v in np.unique(px, axis=0)}
        assert (255, 80, 80) in colors, "객체 색 채움 없음"
        assert (255, 255, 255) in colors, "흰 글씨 없음"

    def test_anchored_bottom_left_above_right_of_point(self):
        """라벨의 **왼쪽 아래**가 기준점(화살표 시작점)에서 오른쪽·위로 살짝 떨어져야 한다."""
        from image_view import LABEL_OFFSET_X, LABEL_OFFSET_Y

        c = blank(h=300, w=800)  # 클램프가 끼어들지 않도록 넉넉하게
        ax, ay = 200, 150
        V._draw_info_label(c, "X: 1.0, Y: 1.0, Deg: 1.0", ax, ay, (0, 200, 0))
        ys, xs = np.nonzero(np.any(c > 0, axis=2))
        assert xs.min() == ax + LABEL_OFFSET_X, "왼쪽 모서리가 기준점 오른쪽이어야"
        assert ys.max() == ay - LABEL_OFFSET_Y, "아래 모서리가 기준점 위여야"
        assert xs.max() > ax and ys.min() < ay  # 오른쪽 위로 뻗어 나감

    def test_clamped_inside_canvas(self):
        """가장자리 객체 — 라벨이 잘려 안 보이는 것보다 안으로 밀어 넣는 편이 낫다."""
        for cx, cy in [(5, 5), (395, 195), (0, 100), (200, 0)]:
            c = blank()
            V._draw_info_label(c, "X: 999.0, Y: 999.0, Deg: 180.0, Open: 1.00", cx, cy, (80, 80, 255))
            ys, xs = np.nonzero(np.any(c > 0, axis=2))
            assert xs.min() >= 0 and xs.max() < c.shape[1], f"({cx},{cy}) 가로 벗어남"
            assert ys.min() >= 0 and ys.max() < c.shape[0], f"({cx},{cy}) 세로 벗어남"

    def test_box_grows_with_text(self):
        """글자 길이에 맞춰 사각형이 커져야 한다 (Open 유무로 길이가 달라짐)."""

        def width(text):
            c = blank()
            V._draw_info_label(c, text, 200, 100, (200, 200, 200))
            xs = np.nonzero(np.any(c > 0, axis=2))[1]
            return xs.max() - xs.min()

        assert width("X: 1.0, Y: 1.0, Deg: 1.0, Open: 0.50") > width("X: 1.0, Y: 1.0, Deg: 1.0")

    def test_numpy_color_accepted(self):
        """팔레트 색이 numpy 정수로 올 수도 있다 — cv2 가 거부하지 않게 int 변환."""
        c = blank()
        V._draw_info_label(c, "X: 1.0", 200, 100, np.array([10, 20, 30], dtype=np.int64))
        assert c.any()


class TestDrawOrder:
    """화살표를 먼저, 정보 라벨을 나중에 그려야 글씨가 가려지지 않는다."""

    def test_label_drawn_over_arrow(self, qapp):
        from image_view import DraggableImageLabel, LABEL_OFFSET_X, LABEL_OFFSET_Y

        v = DraggableImageLabel()
        v.set_image(np.zeros((300, 600, 3), np.uint8))
        center = (300, 200)
        fill = (255, 80, 80)
        arrow_color = (0, 0, 255)
        # 화살표를 라벨이 놓일 자리(중심의 오른쪽 위)로 쏴서 일부러 겹치게 한다
        v.set_obbs([(np.array([[250, 150], [350, 150], [350, 250], [250, 250]]), fill, 0, 12.3, 0.42)])
        v.set_arrows([(center, (560, 60), arrow_color, 0)])
        canvas = v._make_overlay_image()

        # 라벨 사각형 안쪽에 화살표 색이 남아 있으면 안 된다 (= 라벨이 위에 덮였다)
        text_h = 20
        y1 = center[1] - LABEL_OFFSET_Y
        region = canvas[y1 - text_h : y1, center[0] + LABEL_OFFSET_X : center[0] + LABEL_OFFSET_X + 150]
        assert not np.all(region == arrow_color, axis=2).any(), "화살표가 라벨 위에 그려짐"
        assert np.all(region == fill, axis=2).any(), "라벨 채움색이 안 보임"


class TestCenterDot:
    """화살표 시작점(=객체 중심)에 원을 그려 중심 위치가 한눈에 보이게 한다."""

    def test_dot_at_arrow_start(self, qapp):
        from image_view import DraggableImageLabel, CENTER_DOT_R

        v = DraggableImageLabel()
        v.set_image(np.zeros((300, 600, 3), np.uint8))
        center, arrow_color = (300, 200), (0, 0, 255)
        v.set_arrows([(center, (300, 280), arrow_color, 0)])
        canvas = v._make_overlay_image()

        assert tuple(canvas[center[1], center[0]]) == arrow_color, "중심에 원이 없음"
        # 흰 테두리 — 어두운 배경/같은 색 객체 위에서도 원이 묻히지 않게
        assert tuple(canvas[center[1] - CENTER_DOT_R - 2, center[0]]) == (255, 255, 255)  # 흰 테두리 링
        # 원 바깥은 그대로
        assert tuple(canvas[center[1] - CENTER_DOT_R - 4, center[0]]) != arrow_color  # 원 바깥

    def test_dot_survives_arrow_tail(self, qapp):
        """화살표를 먼저, 원을 나중에 그려야 꼬리에 원이 가려지지 않는다."""
        from image_view import DraggableImageLabel

        v = DraggableImageLabel()
        v.set_image(np.zeros((300, 600, 3), np.uint8))
        v.set_arrows([((300, 200), (500, 200), (0, 200, 0), 0)])
        canvas = v._make_overlay_image()
        assert tuple(canvas[200, 300]) == (0, 200, 0)
