"""실물 캡처 회귀 테스트 — 합성 장면만으로는 못 잡는 실패를 고정한다.

배경: 정사각형 화장품 케이스(분홍, 3×3 팬)를 RealSense 로 찍은 실제 이미지에서
여는 방향이 매 실행마다 바뀌고 정반대로도 나오는 문제가 있었다. 원인은 판정 지표가
히스토그램의 **피어슨 상관**이라, 두 색 분포가 겹치지 않으면 "얼마나 먼지"와 무관하게
비슷한 값으로 포화된다는 것 — 네 변 점수가 0.15~0.18 안에 몰려 사실상 동점이 됐다.

이 파일의 fixture 는 그 캡처에서 케이스 부분만 잘라낸 것이다:
  hinge_case.png       (BGR 215×205) — 위쪽 가장자리에 **은색 힌지 바**가 있다
  hinge_case_mask.png  — 케이스 마스크 (그림자 제외)
힌지가 위쪽이므로 **여는 방향은 아래(+y)** 가 정답이다.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opening_analysis import obb_from_mask, opening_from_hinge

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def real_case():
    bgr = cv2.imread(str(DATA / "hinge_case.png"))
    mask = cv2.imread(str(DATA / "hinge_case_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert bgr is not None and mask is not None, "fixture 이미지 없음"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask


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
        res = opening_dir(real_case, band_ratio=band, metric="chroma")
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


class TestOldMetricsWereUnstable:
    """예전 기본값(bg)이 이 사진에서 왜 못 쓰는지 기록 — 기본값을 되돌리지 않기 위한 근거."""

    def test_histogram_metrics_have_tied_scores(self, real_case):
        """bg/odd 는 네 변 점수가 거의 동점이라 노이즈가 답을 정한다."""
        for metric in ("bg", "odd"):
            res = opening_dir(real_case, band_ratio=0.15, metric=metric)
            spread = max(res["scores"]) - min(res["scores"])
            assert spread < 0.12, f"{metric}: 예상과 달리 점수가 벌어짐 {res['scores']}"

    def test_chroma_scores_are_well_separated(self, real_case):
        """반면 chroma 는 같은 사진에서 배수로 갈린다 (띠를 힌지 폭에 맞췄을 때 2.3배)."""
        res = opening_dir(real_case, band_ratio=0.10, metric="chroma")
        srt = sorted(res["scores"], reverse=True)
        assert srt[0] > 2.0 * srt[1], f"{[round(x, 2) for x in res['scores']]}"
