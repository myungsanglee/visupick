"""object_detector.py 순수 헬퍼 테스트 — SAM3 모델/torch 없이 검증 가능한 부분.

배경: "rectangle, circle" 처럼 쉼표로 여러 객체를 적으면 통째로 하나의 명사구로
전달되어 아무것도 못 찾던 버그. 쉼표 분리와 프롬프트 간 중복 제거를 검증한다.
"""

from object_detector import Sam3Detector


class TestSplitPrompts:
    def test_single(self):
        assert Sam3Detector.split_prompts("cosmetic case") == ["cosmetic case"]

    def test_comma_separated(self):
        assert Sam3Detector.split_prompts("rectangle, circle") == ["rectangle", "circle"]

    def test_messy_spacing_and_empties(self):
        assert Sam3Detector.split_prompts(" rectangle ,, circle , ") == ["rectangle", "circle"]

    def test_noun_phrase_with_spaces_kept_whole(self):
        assert Sam3Detector.split_prompts("transparent box, red cap") == ["transparent box", "red cap"]

    def test_empty(self):
        assert Sam3Detector.split_prompts("  ,  ") == []


def det(x1, y1, x2, y2, conf, name):
    return {"bbox": [x1, y1, x2, y2], "confidence": conf, "class_id": 0, "class_name": name}


class TestDedupeOverlaps:
    def test_same_object_two_concepts_keeps_higher_score(self):
        """같은 물체가 두 개념에 다 걸리면 점수 높은 쪽만 남는다."""
        a = det(10, 10, 110, 110, 0.9, "box")
        b = det(12, 11, 112, 111, 0.6, "case")  # 거의 같은 박스
        kept = Sam3Detector.dedupe_overlaps([a, b])
        assert kept == [a]

    def test_distinct_objects_kept(self):
        a = det(0, 0, 50, 50, 0.9, "rectangle")
        b = det(200, 200, 260, 260, 0.8, "circle")
        assert len(Sam3Detector.dedupe_overlaps([a, b])) == 2

    def test_partial_overlap_below_threshold_kept(self):
        """붙어 있는 서로 다른 물체(IoU 낮음)는 지우면 안 된다."""
        a = det(0, 0, 100, 100, 0.9, "rectangle")
        b = det(60, 0, 160, 100, 0.8, "rectangle")  # IoU ≈ 0.25
        assert len(Sam3Detector.dedupe_overlaps([a, b])) == 2
