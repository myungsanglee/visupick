"""
Zivid Camera Module
Zivid 2D/3D 카메라 제어 및 데이터 캡처
"""

import zivid
import zivid.experimental.calibration
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

from base_camera import BaseCamera

logger = logging.getLogger(__name__)


class ZividCamera(BaseCamera):
    """Zivid 카메라 제어 클래스 (BaseCamera 구현체)."""

    def __init__(self):
        self.app = None
        self.camera = None
        self.settings = None
        self._connected = False

    @property
    def is_capture_ready(self) -> bool:
        """Zivid 는 settings(YML) 가 로드돼야 capture() 가 동작."""
        return self.connected and self.settings is not None

    def connect(self) -> bool:
        """카메라 연결"""
        try:
            self.app = zivid.Application()
            cameras = self.app.cameras()
            if not cameras:
                logger.error("연결된 Zivid 카메라가 없습니다")
                return False

            self.camera = cameras[0]
            if not self.camera.state.connected:
                self.camera.connect()

            self._connected = True
            info = self.camera.info
            logger.info(f"Zivid 카메라 연결: {info.model_name} (S/N: {info.serial_number})")
            return True
        except Exception as e:
            logger.error(f"Zivid 카메라 연결 실패: {e}")
            return False

    def disconnect(self):
        """카메라 연결 해제"""
        if self.camera:
            self.camera.disconnect()
            self._connected = False
            logger.info("Zivid 카메라 연결 해제")

    @property
    def connected(self) -> bool:
        return self._connected

    def load_settings(self, yml_path: str) -> bool:
        """
        YML 파일에서 카메라 설정 로드 (Zivid Studio에서 export한 파일)

        Args:
            yml_path: 설정 파일 경로
        """
        try:
            path = Path(yml_path)
            if not path.exists():
                logger.error(f"설정 파일 없음: {yml_path}")
                return False

            self.settings = zivid.Settings.load(path)
            logger.info(f"카메라 설정 로드 완료: {yml_path}")
            return True
        except Exception as e:
            logger.error(f"카메라 설정 로드 실패: {e}")
            return False

    def capture(self) -> Optional[zivid.Frame]:
        """
        3D 캡처 수행

        Returns:
            zivid.Frame 또는 None
        """
        if not self._connected:
            logger.error("카메라가 연결되어 있지 않습니다")
            return None

        if self.settings is None:
            logger.error("카메라 설정이 로드되지 않았습니다")
            return None

        try:
            frame = self.camera.capture(self.settings)
            logger.info("캡처 완료")
            return frame
        except Exception as e:
            logger.error(f"캡처 실패: {e}")
            return None

    @staticmethod
    def frame_to_2d_image(frame: zivid.Frame) -> Optional[np.ndarray]:
        """
        Frame에서 2D 컬러 이미지 추출 (BGR)

        Returns:
            numpy array (H, W, 3) BGR 또는 None
        """
        try:
            point_cloud = frame.point_cloud()
            rgba = point_cloud.copy_data("rgba")
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            return bgr
        except Exception as e:
            logger.error(f"2D 이미지 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_point_cloud(frame: zivid.Frame) -> Optional[np.ndarray]:
        """
        Frame에서 3D 포인트 클라우드 추출

        Returns:
            numpy array (H, W, 3) XYZ (mm) 또는 None
        """
        try:
            point_cloud = frame.point_cloud()
            xyz = point_cloud.copy_data("xyz")
            return xyz
        except Exception as e:
            logger.error(f"포인트 클라우드 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_normals(frame: zivid.Frame) -> Optional[np.ndarray]:
        """
        Frame에서 법선 벡터 추출 (Zivid가 계산한 per-pixel normal)

        Returns:
            numpy array (H, W, 3) 단위 법선 벡터 (카메라 좌표계) 또는 None
        """
        try:
            point_cloud = frame.point_cloud()
            normals = point_cloud.copy_data("normals")
            return normals
        except Exception as e:
            logger.error(f"법선 추출 실패: {e}")
            return None

    def get_intrinsics(self) -> Optional[Dict]:
        """
        카메라 내부 파라미터 가져오기 (현재 설정 기준)

        Returns:
            {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [k1,k2,p1,p2,k3]}
        """
        if not self._connected or not self.camera:
            logger.error("카메라가 연결되어 있지 않습니다")
            return None

        try:
            intr = zivid.experimental.calibration.intrinsics(self.camera, self.settings)
            cm = intr.camera_matrix
            dist = intr.distortion

            return {
                "camera_matrix": [
                    [cm.fx, 0, cm.cx],
                    [0, cm.fy, cm.cy],
                    [0, 0, 1],
                ],
                "dist_coeffs": [
                    dist.k1, dist.k2,
                    dist.p1, dist.p2,
                    dist.k3,
                ],
            }
        except Exception as e:
            logger.error(f"카메라 내부 파라미터 가져오기 실패: {e}")
            return None

    @staticmethod
    def save_point_cloud(frame: zivid.Frame, path: str):
        """포인트 클라우드 저장"""
        frame.save(path)
        logger.info(f"포인트 클라우드 저장: {path}")

    @staticmethod
    def detect_checkerboard(
        image: np.ndarray,
        board_size: Tuple[int, int] = (7, 5),
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        체커보드 검출

        Args:
            image: BGR 이미지
            board_size: 체커보드 내부 코너 수 (가로, 세로)

        Returns:
            (검출 성공 여부, 코너 좌표, 오버레이 이미지)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)

        overlay = image.copy()
        if found:
            # 서브픽셀 정밀도로 코너 보정
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(overlay, board_size, corners, found)

        return found, corners, overlay
