"""
Object Detector Modules
=======================
빈 픽킹 객체 검출기를 카메라(base_camera)처럼 **공통 인터페이스**로 다루기 위한 모듈.
Qt/UI 의존이 전혀 없다 — 순수 추론만 담당하고, 실패는 예외로 올려 호출 측 GUI 가
다이얼로그로 표시한다 (bin_picking_tab 의 _detect / _detect_sam3).

검출 결과 표준 포맷 (detections): 리스트, 각 항목은 dict
    {
      "bbox": [x1, y1, x2, y2],       # 픽셀, xyxy
      "confidence": float,
      "class_id": int,
      "class_name": str,
      "mask": np.ndarray(H, W) bool,  # (선택) seg 모델 / SAM3 만 동봉
    }

detect(image_bgr, conf_thresh, ...) → (detections, infer_ms)
  infer_ms = 이미지 입력 → 결과 수신까지 순수 추론 시간(ms). 모델 로드는 제외(캐싱).

구현체:
  - RFDetrDetector : 외부 detector.py(RF-DETR TensorRT/ONNX 래퍼) 사용. 학습된 클래스만 검출.
  - Sam3Detector   : Meta 공식 SAM 3 — 텍스트(명사구) 프롬프트로 검출기 없이 개념 분할.

두 구현 모두 무거운 모델은 **첫 detect() 때 한 번만 로드**해 재사용한다.
"""

import os
import sys
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectorUnavailable(RuntimeError):
    """검출기 모듈/모델을 불러올 수 없음 (미설치·경로 오류 등). 사용자에게 설치 안내용."""


class DetectorError(RuntimeError):
    """추론 실행 중 오류."""


class ObjectDetector(ABC):
    """검출기 추상 인터페이스. 새 검출기는 이걸 상속해 detect() 를 구현한다."""

    name = "detector"

    def load(self) -> None:
        """무거운 모델을 미리 로드 (선택). detect() 도 내부에서 자동 호출하므로
        굳이 부를 필요는 없지만, GUI 가 '로드 중...' 을 먼저 보여주고 싶을 때 쓴다."""
        self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        """모델이 이미 로드됐는지 (GUI 가 로드 안내 표시 여부 판단용)."""
        return self._is_loaded()

    @abstractmethod
    def _ensure_loaded(self) -> None:
        """모델을 (한 번만) 로드. 실패 시 DetectorUnavailable."""

    @abstractmethod
    def _is_loaded(self) -> bool: ...

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, conf_thresh: float, **kwargs) -> Tuple[List[Dict], float]:
        """BGR 이미지 → (detections, infer_ms). 위 표준 포맷."""


class RFDetrDetector(ObjectDetector):
    """외부 detector.py(RF-DETR TensorRT/ONNX 래퍼) 사용. 학습된 클래스만 검출."""

    name = "RF-DETR"

    def __init__(self, model_path: str, class_names: Dict[int, str], detector_dir: Optional[str] = None):
        self.model_path = model_path
        self.class_names = class_names
        self.detector_dir = detector_dir
        self._impl = None  # 로드된 Detector 인스턴스 (첫 detect 때 생성)

    def _is_loaded(self) -> bool:
        return self._impl is not None

    def _ensure_loaded(self) -> None:
        if self._impl is not None:
            return
        if self.detector_dir and self.detector_dir not in sys.path:
            sys.path.insert(0, self.detector_dir)  # `from detector import Detector` 가능하게
        try:
            from detector import Detector
        except ImportError as e:
            raise DetectorUnavailable(
                f"detector 모듈 import 실패:\n{e}\n\n" f"경로: {self.detector_dir}\n" "필요 패키지: supervision, tensorrt, pycuda"
            )
        if not os.path.exists(self.model_path):
            raise DetectorUnavailable(f"모델 파일 없음:\n{self.model_path}")
        self._impl = Detector(
            model_path=self.model_path,
            model_name="rf-detr",
            class_names=self.class_names,
            conf_thresh=0.5,
        )

    def detect(self, image_bgr: np.ndarray, conf_thresh: float, **kwargs) -> Tuple[List[Dict], float]:
        self._ensure_loaded()
        self._impl.conf_thresh = conf_thresh  # conf 는 매 추론마다 갱신
        try:
            t0 = time.perf_counter()
            _, result = self._impl.predict(image_bgr)
            infer_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            raise DetectorError(f"RF-DETR 추론 실패:\n{e}")

        detections: List[Dict] = []
        xyxy = result["xyxy"]
        confs = result["confidence"]
        class_ids = result["class_id"]
        class_names = result["class_name"]
        masks = result.get("mask")  # seg 모델이면 (N, H, W) bool, det 모델이면 None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            det = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confs[i]),
                "class_id": int(class_ids[i]),
                "class_name": class_names[i],
            }
            if masks is not None and i < len(masks):
                det["mask"] = np.asarray(masks[i], dtype=bool)
            detections.append(det)
        return detections, infer_ms


class Sam3Detector(ObjectDetector):
    """Meta 공식 SAM 3 — 텍스트(명사구) 프롬프트 개념 분할 (검출기 불필요).

    SAM 3 의 set_text_prompt 는 **호출당 개념(명사구) 하나**만 받는다.
    "rectangle, circle" 처럼 쉼표로 여러 객체를 적으면 통째로 하나의 낯선
    명사구가 되어 아무것도 못 찾는다. 그래서 detect() 가 쉼표를 다중 프롬프트로
    해석해 개념마다 따로 추론한다 — 무거운 이미지 임베딩(set_image)은 한 번만
    계산하고 state 를 재사용하므로 개념이 늘어도 비용은 크게 늘지 않는다.
    """

    name = "SAM3"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._processor = None  # Sam3Processor (첫 detect 때 로드)

    @staticmethod
    def _to_numpy(x):
        """torch.Tensor / list / ndarray 무엇이 오든 numpy 로 통일 (SAM3 출력 대비).

        autocast 때문에 텐서가 bfloat16/float16 으로 나올 수 있는데 numpy 는
        bfloat16 을 지원 안 하므로("unsupported ScalarType BFloat16"), 축소정밀
        부동소수는 float32 로 승격한 뒤 변환한다.
        """
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu()
            if "float16" in str(x.dtype) or "bfloat16" in str(x.dtype):
                x = x.float()  # → float32 (numpy 호환)
            x = x.numpy()
        return np.asarray(x)

    @staticmethod
    def split_prompts(prompt: str) -> List[str]:
        """쉼표 구분 텍스트 → 개념 프롬프트 목록. "rectangle, circle" → ["rectangle", "circle"]."""
        return [p.strip() for p in prompt.split(",") if p.strip()]

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    @classmethod
    def dedupe_overlaps(cls, detections: List[Dict], iou_thresh: float = 0.85) -> List[Dict]:
        """프롬프트 간 중복 제거 — 같은 물체가 두 개념에 다 걸리면("box"와 "case" 등)
        점수 높은 쪽만 남긴다. 안 지우면 한 물체에 픽 후보가 2개 생겨 헷갈린다."""
        kept: List[Dict] = []
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            if all(cls._bbox_iou(det["bbox"], k["bbox"]) < iou_thresh for k in kept):
                kept.append(det)
        return kept

    def _is_loaded(self) -> bool:
        return self._processor is not None

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return
        if self.model_dir and self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)  # repo 가 pip 경로에 없을 때
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise DetectorUnavailable(
                f"SAM3 import 실패:\n{e}\n\n"
                "Meta 공식 repo(facebookresearch/sam3)를 설치하세요.\n"
                "repo 가 pip 경로에 없으면 환경변수 SAM3_MODEL_DIR 로 repo 루트를 지정."
            )
        self._processor = Sam3Processor(build_sam3_image_model())

    def detect(self, image_bgr: np.ndarray, conf_thresh: float, prompt: str = "", **kwargs) -> Tuple[List[Dict], float]:
        if not prompt:
            raise DetectorError("검출할 객체를 설명하는 텍스트를 입력하세요 (예: cosmetic case)")
        self._ensure_loaded()
        try:
            import torch
            from PIL import Image
        except ImportError as e:
            raise DetectorUnavailable(f"SAM3 의존 패키지(torch/PIL) import 실패:\n{e}")

        prompts = self.split_prompts(prompt)
        if not prompts:
            raise DetectorError("검출할 객체를 설명하는 텍스트를 입력하세요 (예: cosmetic case)")

        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # SAM3 는 RGB PIL 기대
            # SAM3 가중치는 bf16 → 입력(float32)과 dtype 불일치 방지 위해 공식 예제처럼
            # autocast(bfloat16) 컨텍스트에서 추론 (프로세서 내부엔 autocast 없음).
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # 추론 시간: 이미지 입력 → 결과(numpy) 수신까지. _to_numpy 의 .cpu()가 GPU
            # 커널을 동기화하므로 여기까지가 실제 "결과 받은 시점" (모델 로드는 제외).
            t0 = time.perf_counter()
            per_prompt = []  # [(개념 텍스트, masks, boxes, scores)]
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                state = self._processor.set_image(Image.fromarray(rgb))  # 임베딩 1회
                for p in prompts:
                    output = self._processor.set_text_prompt(state=state, prompt=p)
                    per_prompt.append((p, self._to_numpy(output["masks"]), self._to_numpy(output["boxes"]), self._to_numpy(output["scores"])))
            infer_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            raise DetectorError(f"SAM3 추론 실패:\n{e}")

        detections: List[Dict] = []
        for cls_id, (p, masks, boxes, scores) in enumerate(per_prompt):
            n = 0 if boxes is None else len(boxes)
            for i in range(n):
                score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                if score < conf_thresh:  # Conf 를 점수 임계값으로 재사용
                    continue
                x1, y1, x2, y2 = [float(v) for v in np.asarray(boxes[i]).ravel()[:4]]
                det = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "class_id": cls_id,
                    "class_name": p,
                }
                if masks is not None and i < len(masks):
                    m = np.squeeze(np.asarray(masks[i]))  # (1,H,W)/(H,W), bool 또는 logit
                    if m.ndim == 2:
                        det["mask"] = m if m.dtype == bool else (m > 0.5)
                detections.append(det)

        if len(per_prompt) > 1:
            detections = self.dedupe_overlaps(detections)
        return detections, infer_ms
