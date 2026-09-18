"""opening_analysis.opening_from_hinge 테스트 — 정사각형 케이스 4방향 힌지 판별.

합성 장면: 바닥 위에 정사각형 케이스. 한 변 안쪽 띠만 '투명'(바닥이 비쳐 보임)이고
나머지는 불투명한 제품 색. 알고리즘이 그 변을 힌지로 골라 **반대 방향**을 여는
방향으로 주는지 확인한다. 방향은 box_pts 순서에 의존하지 않도록 hinge_side 인덱스가
아니라 **벡터 부호**로 검증한다.
"""

import cv2
import numpy as np
import pytest

from opening_analysis import obb_from_mask, opening_from_hinge

H, W = 400, 400
FLOOR = (185, 180, 170)  # 밝은 회백색 바닥
PRODUCT = (70, 60, 110)  # 어두운 제품(불투명)


def scene(hinge_side="top", angle=0.0, size=180, band=0.25, seed=0):
    """(rgb, mask) 생성. hinge_side 안쪽 띠는 바닥색 = 비쳐 보이는 투명부."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = FLOOR
    rgb = np.clip(rgb + rng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)

    # 케이스를 축정렬로 그린 뒤 통째로 회전 → 회전해도 같은 물리 구성
    case = np.zeros((H, W, 3), np.uint8)
    case[:] = PRODUCT
    m = np.zeros((H, W), np.uint8)
    half = size // 2
    y0, y1 = H // 2 - half, H // 2 + half
    x0, x1 = W // 2 - half, W // 2 + half
    m[y0:y1, x0:x1] = 1
    t = int(size * band)
    sl = {
        "top": (slice(y0, y0 + t), slice(x0, x1)),
        "bottom": (slice(y1 - t, y1), slice(x0, x1)),
        "left": (slice(y0, y1), slice(x0, x0 + t)),
        "right": (slice(y0, y1), slice(x1 - t, x1)),
    }[hinge_side]
    case[sl] = FLOOR  # 투명부: 바닥이 그대로 비쳐 보임
    case = np.clip(case + rng.normal(0, 6, case.shape), 0, 255).astype(np.uint8)

    if angle:
        R = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        case = cv2.warpAffine(case, R, (W, H))
        m = cv2.warpAffine(m, R, (W, H), flags=cv2.INTER_NEAREST)
    out = rgb.copy()
    out[m > 0] = case[m > 0]
    return out, m.astype(bool)


def expected_dir(hinge_side):
    """힌지 반대쪽 = 여는 방향 (이미지 좌표: y 아래가 +)."""
    return {"top": (0, 1), "bottom": (0, -1), "left": (1, 0), "right": (-1, 0)}[hinge_side]


def run(rgb, mask, **kw):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return opening_from_hinge(mask, gray, obb_from_mask(mask), rgb=rgb, **kw)


class TestHingeSideDetection:
    @pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
    def test_each_side(self, side):
        """네 변 어디에 힌지가 있어도 반대 방향을 가리켜야 한다."""
        res = run(*scene(hinge_side=side))
        assert res is not None
        ex, ey = expected_dir(side)
        dx, dy = res["dir"]
        assert dx * ex + dy * ey > 0.9, f"{side}: dir={res['dir']} 기대≈{(ex, ey)}"
        assert res["confidence"] > 0.15

    @pytest.mark.parametrize("angle", [20.0, 45.0, 70.0])
    def test_rotated_case(self, angle):
        """케이스가 회전해도(=OBB 축이 바뀌어도) 물리적 여는 방향은 같아야 한다."""
        res = run(*scene(hinge_side="top", angle=angle))
        assert res is not None
        # 힌지 띠도 같이 회전했으므로 기대 방향도 회전
        a = np.radians(-angle)  # 이미지 y축이 아래 → 회전 부호 반전
        ex, ey = np.cos(a) * 0 - np.sin(a) * 1, np.sin(a) * 0 + np.cos(a) * 1
        dx, dy = res["dir"]
        assert dx * ex + dy * ey > 0.85, f"{angle}°: dir={res['dir']} 기대≈{(ex, ey)}"

    def test_gray_only_still_works(self):
        """컬러가 없어도(gray 만) 밝기 분포 차이로 판별 가능해야 한다."""
        rgb, mask = scene(hinge_side="right")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        res = opening_from_hinge(mask, gray, obb_from_mask(mask))
        assert res is not None
        dx, dy = res["dir"]
        assert dx < -0.9  # 오른쪽이 힌지 → 왼쪽(-x)이 여는 방향


class TestMetrics:
    def test_odd_metric(self):
        """이질도 방식도 같은 변을 힌지로 골라야 한다."""
        res = run(*scene(hinge_side="bottom"), metric="odd")
        assert res is not None
        assert res["dir"][1] < -0.9  # 아래가 힌지 → 위(-y)

    def test_uniform_case_low_confidence(self):
        """네 변이 똑같으면(투명부 없음) 신뢰도가 낮아야 한다 — 틀린 방향을 확신하지 않게."""
        rgb, mask = scene(hinge_side="top", band=0.0)  # 띠 두께 0 = 균일
        res = run(rgb, mask)
        assert res is None or res["confidence"] < 0.5

    def test_scores_reported(self):
        res = run(*scene(hinge_side="left"))
        assert len(res["scores"]) == 4
        assert res["hinge_side"] == int(np.argmax(res["scores"]))


class TestGuards:
    def test_tiny_object(self):
        m = np.zeros((H, W), bool)
        m[10:20, 10:20] = True
        rgb = np.zeros((H, W, 3), np.uint8)
        assert opening_from_hinge(m, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), obb_from_mask(m), rgb=rgb) is None

    def test_band_ratio_clamped(self):
        """0.5 이상을 넣어도 터지지 않고 클램프돼야 한다 (띠가 겹치면 판별 불가)."""
        res = run(*scene(hinge_side="top"), band_ratio=0.9)
        assert res is not None


class TestConfidenceIsHonest:
    """신뢰도는 1·2등의 **절대** 격차여야 한다.

    예전에는 전체 산포로 나눈 상대 격차라, 네 변이 거의 동점이어도(마스크가 투명한
    힌지부를 못 덮어 신호가 없는 경우) 비율만 크면 conf 가 0.9 를 넘었다 — "동점인데
    확신하는" 위험. 회귀 방지용.
    """

    def _no_hinge_scene(self, seed):
        """마스크가 케이스 본체만 덮어 네 띠가 전부 같은 색 — 판별 근거가 없는 상황."""
        rng = np.random.default_rng(seed)
        rgb = np.full((H, W, 3), 0, np.uint8)
        rgb[:] = FLOOR
        m = np.zeros((H, W), np.uint8)
        y0, y1, x0, x1 = 130, 290, 130, 290
        rgb[y0:y1, x0:x1] = PRODUCT
        j = rng.integers(-2, 3, size=4)
        m[y0 + j[0] : y1 + j[1], x0 + j[2] : x1 + j[3]] = 1
        rgb = np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)
        return rgb, m.astype(bool)

    @pytest.mark.parametrize("metric", ["bg", "odd"])
    def test_tied_sides_report_low_confidence(self, metric):
        for seed in range(6):
            res = run(*self._no_hinge_scene(seed), metric=metric)
            if res is None:
                continue
            assert res["confidence"] < 0.15, f"근거 없는데 conf={res['confidence']:.2f} (seed={seed})"

    def test_clear_hinge_reports_high_confidence(self):
        res = run(*scene(hinge_side="top"))
        assert res["confidence"] > 0.3

    def test_confidence_equals_top_two_gap(self):
        res = run(*scene(hinge_side="left"))
        srt = sorted(res["scores"], reverse=True)
        assert res["confidence"] == pytest.approx(min(1.0, srt[0] - srt[1]), abs=1e-6)
