"""실물 캡처 회귀 테스트 — 합성 장면만으로는 못 잡는 실패를 고정한다.

배경: 정사각형 화장품 케이스(분홍, 3×3 팬)를 RealSense 로 찍은 실제 이미지에서
여는 방향이 매 실행마다 바뀌고 정반대로도 나오는 문제가 있었다. 원인은 판정 지표가
히스토그램의 **피어슨 상관**이라, 두 색 분포가 겹치지 않으면 "얼마나 먼지"와 무관하게
비슷한 값으로 포화된다는 것 — 네 변 점수가 0.15~0.18 안에 몰려 사실상 동점이 됐다.

이 파일의 fixture 는 실제 캡처에서 케이스 부분만 잘라낸 것이다. 둘 다 **힌지가 위쪽**
이므로 **여는 방향은 아래(+y)** 가 정답이고, 서로 다른 단서로 풀어야 한다:

  hinge_case.png / _mask.png       — 분홍 케이스. 위 가장자리에 **은색 힌지 바**.
      본체가 분홍이라 **색(a*b*)** 으로 갈린다. 밝기는 요철 그늘에 속아 왼쪽을 고른다.
  hinge_case_dark.png / _mask.png  — 거의 검은 케이스(rom&nd). 위 가장자리가 **투명해
      밝다**. 본체가 무채색이라 색으로는 신호가 없고 **밝기(L)** 로 갈린다. 게다가 아래쪽에
      흰색 로고가 인쇄돼 있어, 색만 보면 그 변이 1등이 되어 **정반대**가 나왔다.

그래서 알고리즘은 두 단서를 모두 계산해 **1·2등 격차가 큰 쪽을 자동 채택**한다.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opening_analysis import obb_from_mask, opening_from_hinge

DATA = Path(__file__).parent / "data"


def _load(stem):
    bgr = cv2.imread(str(DATA / f"{stem}.png"))
    mask = cv2.imread(str(DATA / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert bgr is not None and mask is not None, f"fixture 없음: {stem}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask > 127


@pytest.fixture(scope="module")
def real_case():
    return _load("hinge_case")


@pytest.fixture(scope="module")
def dark_case():
    return _load("hinge_case_dark")


def opening_dir(real_case, **kw):
    rgb, gray, mask = real_case
    res = opening_from_hinge(mask, gray, obb_from_mask(mask), rgb=rgb, **kw)
    assert res is not None
    return res


class TestRealCapture:
    @pytest.mark.parametrize("band", [0.10, 0.15, 0.20, 0.25, 0.30])
    def test_chroma_finds_hinge_at_every_band(self, real_case, band):
        """기본 방식(chroma)은 띠 두께와 무관하게 아래(+y)를 가리켜야 한다.

        예전 방식(bg/odd)은 같은 사진에서 설정에 따라 위/오른/왼으로 튀었다.
        """
        res = opening_dir(real_case, band_ratio=band)
        dx, dy = res["dir"]
        assert dy > 0.9, f"band={band}: dir={res['dir']} (아래를 가리켜야 함)"

    def test_narrow_band_gives_stronger_confidence(self, real_case):
        """띠를 실제 힌지 폭(≈8%)에 가깝게 잡을수록 신뢰도가 높아야 한다."""
        narrow = opening_dir(real_case, band_ratio=0.10)["confidence"]
        wide = opening_dir(real_case, band_ratio=0.30)["confidence"]
        assert narrow > wide

    def test_hinge_side_is_one_specific_side(self, real_case):
        """네 점수 중 힌지 쪽 하나만 뚜렷이 커야 한다 (동점이면 찍는 것)."""
        res = opening_dir(real_case, band_ratio=0.15)
        srt = sorted(res["scores"], reverse=True)
        assert srt[0] > srt[1] * 1.5, f"1등이 2등과 비슷함: {[round(x, 2) for x in res['scores']]}"


class TestScoreSeparation:
    """점수가 확실히 갈려야 한다 — 예전 지표(히스토그램 상관)는 이 사진에서 네 변이
    0.15~0.18 안에 몰려 동점이었고, 그래서 정지 장면인데도 답이 매번 바뀌었다.
    (그 지표들은 제거됨 — opening_analysis 모듈 상단 '제거된 대안' 주석 참고.)"""

    def test_scores_are_well_separated(self, real_case):
        """띠를 힌지 폭에 맞추면 1등이 2등의 2배 이상으로 갈린다."""
        res = opening_dir(real_case, band_ratio=0.10)
        srt = sorted(res["scores"], reverse=True)
        assert srt[0] > 2.0 * srt[1], f"{[round(x, 2) for x in res['scores']]}"


class TestDarkCase:
    """무채색(거의 검은) 케이스 — 색으로는 신호가 없고 **밝기**로 갈려야 한다.

    아래쪽에 흰 로고가 인쇄돼 있어, 색만 보면 그 변이 1등이 되어 여는 방향이 **정반대**
    로 나왔던 실제 사례.
    """

    @pytest.mark.parametrize("band", [0.08, 0.10, 0.15, 0.20, 0.25, 0.30])
    def test_finds_hinge_at_every_band(self, dark_case, band):
        res = opening_dir(dark_case, band_ratio=band)
        dx, dy = res["dir"]
        assert dy > 0.9, f"band={band}: dir={res['dir']} (아래를 가리켜야 함)"

    def test_uses_lightness_cue(self, dark_case):
        """본체가 무채색이라 색이 아니라 밝기로 판별해야 한다."""
        assert opening_dir(dark_case, band_ratio=0.15)["feature"] == "lightness"

    def test_confident(self, dark_case):
        assert opening_dir(dark_case, band_ratio=0.15)["confidence"] > 0.5


class TestCuePerCase:
    """제품에 따라 다른 단서가 자동 선택되어야 한다 — 이게 이 알고리즘의 핵심."""

    def test_pink_uses_color(self, real_case):
        assert opening_dir(real_case, band_ratio=0.15)["feature"] == "color"

    def test_dark_uses_lightness(self, dark_case):
        assert opening_dir(dark_case, band_ratio=0.15)["feature"] == "lightness"
