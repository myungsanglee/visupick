"""
Base Camera Interface
=====================
3D 카메라(Zivid, RealSense 등)를 본 시스템에서 동일한 인터페이스로 다루기 위한
추상 클래스. 새 카메라를 추가하려면 BaseCamera 를 상속해 모든 추상 메서드를 구현.

설계 원칙:
  - "프레임"은 카메라 SDK 의 원본 객체 (Zivid Frame, RealSense Frameset 등).
    `frame_to_*` 정적 메서드들이 이를 표준 형식으로 변환한다.
  - 모든 3D 좌표는 mm 단위 (카메라 좌표계).
  - 2D 이미지는 BGR (OpenCV 기본).
  - 내부 파라미터는 OpenCV 표준 dict 포맷.

검증된 구현:
  - ZividCamera (zivid_camera.py)
  - RealSenseCamera (realsense_camera.py)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class BaseCamera(ABC):
    """3D 카메라 추상 인터페이스."""

    # ---------------------------------------------------------------
    # 인스턴스 상태 (서브클래스 __init__에서 초기화 — 컨벤션)
    #   self._connected: bool   (connect 성공 시 True, disconnect 시 False)
    #   self.settings:   Any    (load_settings 성공 시 채워짐, 미로드는 None)
    # 본 시스템 코드는 self.camera.connected, self.camera.settings 형태로
    # 두 속성을 직접 참조하므로 서브클래스는 반드시 둘 다 가져야 한다.
    # ---------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """카메라가 현재 연결돼 있는지."""
        return getattr(self, "_connected", False)

    @property
    def is_capture_ready(self) -> bool:
        """
        캡처 준비 완료 여부. 기본 구현은 'connected' 만 확인.
        서브클래스가 추가 조건이 있으면 override.

        예) ZividCamera: 연결 + settings 가 None 이 아닐 때만 True
            RealSenseCamera: 연결만 되면 True (settings 는 선택적)
        """
        return self.connected

    # ---------------------------------------------------------------
    # 연결 / 설정
    # ---------------------------------------------------------------
    @abstractmethod
    def connect(self) -> bool:
        """카메라 연결. 성공 시 True."""

    @abstractmethod
    def disconnect(self) -> None:
        """카메라 연결 해제 (재호출 안전해야 함)."""

    @abstractmethod
    def load_settings(self, path: str) -> bool:
        """YML/JSON 등 설정 파일 로드. 성공 시 True.
        SDK 마다 설정 포맷이 다르지만 외부 인터페이스는 동일."""

    # ---------------------------------------------------------------
    # 캡처
    # ---------------------------------------------------------------
    @abstractmethod
    def capture(self) -> Any:
        """한 프레임 캡처. SDK 의 원본 프레임 객체 반환 (또는 None on failure).
        반환된 객체는 아래 `frame_to_*` 메서드 입력으로 사용."""

    # ---------------------------------------------------------------
    # 프레임 → 표준 데이터 변환 (정적, 어느 카메라든 같은 출력 형식)
    # ---------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def frame_to_2d_image(frame: Any) -> Optional[np.ndarray]:
        """frame → (H, W, 3) BGR uint8 이미지. 실패 시 None."""

    @staticmethod
    @abstractmethod
    def frame_to_point_cloud(frame: Any) -> Optional[np.ndarray]:
        """frame → (H, W, 3) float32 XYZ 포인트 클라우드 (mm 단위, 카메라 좌표계).
        깊이가 없는 픽셀은 NaN. 실패 시 None."""

    @staticmethod
    @abstractmethod
    def frame_to_normals(frame: Any) -> Optional[np.ndarray]:
        """frame → (H, W, 3) float32 표면 법선 (단위 벡터). SDK 가 제공 안 하면 None.
        본 시스템은 None 받으면 호출 측이 점군에서 자체 추정함."""

    @abstractmethod
    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """현재 설정 기준 카메라 내부 파라미터.
        반환 형식:
            {
              "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
              "dist_coeffs":   [k1,k2,p1,p2,k3],
            }
        """

    # ---------------------------------------------------------------
    # 파일 저장 (선택)
    # ---------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def save_point_cloud(frame: Any, path: str) -> None:
        """원본 포인트 클라우드를 SDK 네이티브 포맷으로 저장 (재처리용).
        파일 확장자는 SDK 마다 다름 (.zdf for Zivid, .ply for RealSense)."""

    # ---------------------------------------------------------------
    # 공통 유틸 (서브클래스 공유 — 추상 아님, 그대로 사용 가능)
    # ---------------------------------------------------------------
    @staticmethod
    def detect_checkerboard(
        image: np.ndarray,
        board_size: Tuple[int, int] = (7, 5),
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """체커보드 코너 검출 — 카메라 종속 X (OpenCV).
        Returns (found, corners[N,1,2], overlay_image).
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)

        overlay = image.copy()
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(overlay, board_size, corners, found)
        return found, corners, overlay
