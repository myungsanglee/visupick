"""opening_analysis.py 테스트 — 합성 마스크/이미지로 OBB·여는 방향을 검증한다.

실제 카메라 없이도 알고리즘 회귀를 잡기 위한 테스트: 회전된 직사각형 마스크의
OBB 복원 정확도, 그리고 '내부 격자 비대칭' 방식이 밝은 띠(여는 쪽 힌지 여백)를
올바른 쪽으로 찾는지 확인한다.
"""

import cv2
import numpy as np
import pytest

from opening_analysis import obb_from_mask, opening_from_grid, opening_weight_map, opening_from_weight


def rect_mask(w=640, h=480, cx=320, cy=240, rw=200, rh=100, angle=0.0):
    """중심 (cx,cy), 크기 (rw,rh), 회전 angle° 의 직사각형 마스크."""
    mask = np.zeros((h, w), np.uint8)
    box = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    return mask.astype(bool)


class TestObbFromMask:
    def test_axis_aligned(self):
        obb = obb_from_mask(rect_mask())
        assert obb is not None
        cx, cy = obb["center"]
        assert (cx, cy) == pytest.approx((320, 240), abs=1.5)
        long_side = max(obb["size"])
        short_side = min(obb["size"])
        assert long_side == pytest.approx(200, abs=3)
        assert short_side == pytest.approx(100, abs=3)

    @pytest.mark.parametrize("angle", [15.0, 45.0, 75.0])
    def test_rotated(self, angle):
        """회전해도 변 길이는 보존돼야 한다."""
        obb = obb_from_mask(rect_mask(angle=angle))
        assert obb is not None
        assert max(obb["size"]) == pytest.approx(200, abs=4)
        assert min(obb["size"]) == pytest.approx(100, abs=4)

    def test_empty_mask(self):
        assert obb_from_mask(np.zeros((100, 100), bool)) is None


class TestOpeningFromGrid:
    def _case(self, margin_side):
        """투명 케이스 흉내: 내부에 세로 칸막이(격자 에지)가 있는 밴드가 한쪽으로
        치우쳐 있고, margin_side 쪽 긴 변에 빈 여백이 남는다.

        opening_from_grid 는 |gx|(세로 에지) 행 프로파일로 격자 밴드를 찾고,
        밴드 위/아래 여백 중 **넓은 쪽**을 여는 방향으로 잡는다.
        """
        mask = rect_mask(cx=320, cy=240, rw=300, rh=150)
        gray = np.full((480, 640), 60, np.uint8)
        gray[mask] = 110
        # 격자 밴드: 여백 반대쪽으로 치우친 세로 칸막이 5개 (2×5 격자 흉내)
        band_cy = 240 + 35 if margin_side == "top" else 240 - 35
        y0, y1 = band_cy - 30, band_cy + 30
        for k in range(6):
            x = 320 - 125 + k * 50
            gray[y0:y1, x - 1 : x + 2] = 240  # 밝은 세로 칸막이 → 강한 |gx|
        return mask, gray

    @pytest.mark.parametrize("side,expect_dy_sign", [("top", -1), ("bottom", +1)])
    def test_margin_side_detected(self, side, expect_dy_sign):
        mask, gray = self._case(side)
        obb = obb_from_mask(mask)
        res = opening_from_grid(mask, gray, obb)
        assert res is not None
        dx, dy = res["dir"]
        assert abs(dy) > abs(dx), "여는 방향은 긴 변에 수직(세로)이어야 함"
        assert np.sign(dy) == expect_dy_sign
        assert res["confidence"] > 0

    def test_no_grid_returns_none(self):
        """세로 에지가 전혀 없는 평평한 케이스는 격자 방식이 방향을 내지 않아야 한다
        (틀린 방향을 자신 있게 내는 것보다 None 이 안전)."""
        mask = rect_mask(cx=320, cy=240, rw=300, rh=150)
        gray = np.full((480, 640), 110, np.uint8)
        res = opening_from_grid(mask, gray, obb_from_mask(mask))
        assert res is None or res["confidence"] < 0.1


class TestOpeningFromWeight:
    def test_bright_side(self):
        """'밝기' 가중치 방식도 밝은 쪽 긴 변을 가리켜야 한다."""
        mask = rect_mask(cx=320, cy=240, rw=300, rh=150)
        gray = np.full((480, 640), 60, np.uint8)
        gray[mask] = 100
        band = rect_mask(cx=320, cy=240 - 60, rw=300, rh=25)
        gray[band] = 240
        obb = obb_from_mask(mask)
        weight = opening_weight_map(gray, "brightness", 70)
        res = opening_from_weight(mask, weight, obb)
        assert res is not None
        dx, dy = res["dir"]
        assert abs(dy) > abs(dx) and dy < 0
