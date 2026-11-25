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


# ── GPU 메모리 부족 안내 ─────────────────────────────────────────────
# 배경: 같은 PC 의 다른 사용자 작업(vLLM)이 GPU 를 선점해 SAM3 로드가 CUDA OOM 으로
# 죽었는데, 원시 traceback 만 떠서 원인을 알 수 없었다.
import pytest

from object_detector import DetectorError, DetectorOutOfMemory, format_gpu_oom_message, is_gpu_oom, raise_if_gpu_oom


class OutOfMemoryError(RuntimeError):
    """torch.OutOfMemoryError 흉내 (이름으로 판별하므로 torch 불필요)."""


class TestGpuOom:
    def test_detects_torch_oom_by_type_name(self):
        assert is_gpu_oom(OutOfMemoryError("whatever"))

    def test_detects_by_message(self):
        assert is_gpu_oom(RuntimeError("CUDA out of memory. Tried to allocate 42.00 MiB"))
        assert is_gpu_oom(RuntimeError("cuMemAlloc failed: CUDA_ERROR_OUT_OF_MEMORY"))

    def test_other_errors_pass_through(self):
        assert not is_gpu_oom(ValueError("shape mismatch"))
        raise_if_gpu_oom(ValueError("shape mismatch"), "SAM3", "추론")  # 아무 일 없어야 함

    def test_oom_is_detector_error(self):
        """기존 `except DetectorError` 핸들러가 그대로 잡아야 한다."""
        with pytest.raises(DetectorError) as ei:
            raise_if_gpu_oom(OutOfMemoryError("x"), "SAM3", "로드")
        assert isinstance(ei.value, DetectorOutOfMemory)
        assert ei.value.title == "GPU 메모리 부족"

    def test_message_names_other_processes(self):
        gpus = [{"index": 0, "name": "RTX A6000", "total_mib": 48530, "free_mib": 33}]
        procs = [
            {"pid": 100, "used_mib": 3256, "name": "python", "user": "robotegra"},
            {"pid": 200, "used_mib": 44912, "name": "VLLM::EngineCore", "user": "lab"},
        ]
        msg = format_gpu_oom_message("SAM3", "로드", gpus, procs, own_pid=100)
        assert "SAM3 모델을 로드하지 못했습니다" in msg
        assert "여유 0.0 GiB" in msg
        # 많이 쓰는 순 → 다른 사용자 프로세스가 먼저, 이 앱은 표시
        assert msg.index("PID 200 [lab] VLLM::EngineCore — 43.9 GiB") < msg.index("PID 100")
        assert "PID 100 [robotegra] python — 3.2 GiB  ← 이 프로그램" in msg
        assert "다른 프로세스가 끝나기를" in msg

    def test_message_without_nvidia_smi(self):
        msg = format_gpu_oom_message("RF-DETR", "추론", [], [], own_pid=1)
        assert "확인하지 못했습니다" in msg
        assert "다른 프로세스가 끝나기를" not in msg
