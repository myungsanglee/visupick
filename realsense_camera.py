"""
Intel RealSense Camera Module (D415 검증, D435/L515 도 호환)
============================================================
BaseCamera 인터페이스 구현 — Zivid 와 동일한 외부 API.

설계 결정:
  - depth → color 정렬 (`rs.align(rs.stream.color)`) → color 픽셀과 같은 좌표계의
    XYZ 가 나옴. 본 시스템의 모든 `(H, W, 3)` 포인트 클라우드 가정과 호환.
  - 깊이 단위: librealsense 의 raw depth 는 mm 가 아니라 16-bit 정수 (depth_scale
    적용 필요). 본 모듈은 mm 로 변환해서 반환 — Zivid 와 단위 통일.
  - IR Emitter: 캡처 시 매번 켜고 끄지 않음 (호출 측이 calibration 캡처용으로
    잠시 끄고 싶으면 `set_emitter_enabled(False)` 호출).
  - 설정 파일: D415 는 JSON (RealSense Viewer 에서 export). YML 도 시도하지만
    JSON 이 표준이라 권장.

알려진 한계:
  - 깊이 정밀도 2–5 mm @ 1 m (Zivid 의 ~30 배). Hand-eye calibration 정확도가
    Zivid 대비 떨어지므로 `compute_hand_eye(prefer_pnp=True)` 사용 권장.
  - 텍스처 없는 평면/반사면에서 깊이 hole 발생 → NaN 픽셀 많아짐.
  - 표면 법선 직접 제공 안 함 → frame_to_normals 는 None 반환.
    호출 측이 점군에서 자체 추정 (본 시스템 코드는 이미 fallback 보유).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from base_camera import BaseCamera

logger = logging.getLogger(__name__)

# pyrealsense2 는 시스템 패키지가 필요한 경우가 있어 import 실패 가능.
# 모듈 로드 실패 시 RealSenseCamera 인스턴스화 시점에 에러를 내도록 처리.
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except Exception as e:
    rs = None
    HAS_REALSENSE = False
    _IMPORT_ERROR = str(e)


# RealSense frame 표현: 본 모듈에서는 dict 로 묶어서 전달
# (Zivid 와 달리 SDK frame 이 즉시 사라지지 않으므로 dict 가 더 안전).
RSFrame = Dict[str, Any]


class RealSenseCamera(BaseCamera):
    """Intel RealSense (D415 검증) 카메라 제어. BaseCamera 구현체."""

    # 기본 해상도. D415 의 depth + color 동시 스트림이 안정적으로 동작하는 조합.
    # (1280x720@30 은 환경/펌웨어에 따라 'Couldn't resolve requests' 발생 사례가 있어
    #  640x480@30 을 표준으로 고정.)
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30

    def __init__(self):
        if not HAS_REALSENSE:
            raise RuntimeError(
                f"pyrealsense2 import 실패: {_IMPORT_ERROR}\n"
                "설치: pip install pyrealsense2 (Linux 는 librealsense SDK 도 필요)"
            )
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)  # depth → color frame 좌표계 정렬
        self.profile = None
        self.depth_scale = 0.001  # raw depth → meter (D415 기본). connect 시 갱신.
        self.settings: Optional[Dict[str, Any]] = None
        self._connected = False

    # ---------------------------------------------------------------
    # 연결 / 설정
    # ---------------------------------------------------------------
    def connect(self) -> bool:
        try:
            self.config.enable_stream(
                rs.stream.depth,
                self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT,
                rs.format.z16, self.DEFAULT_FPS,
            )
            self.config.enable_stream(
                rs.stream.color,
                self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT,
                rs.format.bgr8, self.DEFAULT_FPS,
            )
            self.profile = self.pipeline.start(self.config)

            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())  # raw → meter
            self._connected = True

            # 첫 몇 프레임은 노이즈 큰 경향 → auto-exposure 안정화
            for _ in range(5):
                self.pipeline.wait_for_frames()

            logger.info(
                f"RealSense 연결 성공 ({self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT} "
                f"@ {self.DEFAULT_FPS}fps, depth_scale={self.depth_scale} m/raw)"
            )
            return True
        except Exception as e:
            logger.error(f"RealSense 연결 실패: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._connected:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(f"RealSense stop 중 오류 (무시): {e}")
            self._connected = False
            logger.info("RealSense 연결 해제")

    def load_settings(self, path: str) -> bool:
        """
        RealSense 는 JSON 사전 (RealSense Viewer Advanced Mode 에서 export) 표준.
        Zivid YML 과의 인터페이스 통일을 위해 같은 메서드명을 사용 — 파일이
        .json 이면 advanced_mode 로 적용, .yml 은 무시 (None 으로 둠).
        """
        try:
            ext = Path(path).suffix.lower()
            if ext == ".json":
                with open(path) as f:
                    payload = json.load(f)
                # RealSense Advanced Mode 로 적용
                if not rs.rs400_advanced_mode(self.profile.get_device()).is_enabled():
                    rs.rs400_advanced_mode(self.profile.get_device()).toggle_advanced_mode(True)
                advnc_mode = rs.rs400_advanced_mode(self.profile.get_device())
                advnc_mode.load_json(json.dumps(payload))
                self.settings = payload
                logger.info(f"RealSense 설정 로드 완료 (JSON): {path}")
                return True
            else:
                # YML 등 다른 포맷: 본 시스템 흐름 호환용으로 settings 만 placeholder 설정
                self.settings = {"_path": path, "_note": "non-JSON settings — defaults used"}
                logger.info(f"RealSense: non-JSON 설정 — 기본값 사용 ({path})")
                return True
        except Exception as e:
            logger.error(f"RealSense 설정 로드 실패: {e}")
            return False

    # ---------------------------------------------------------------
    # IR Emitter 제어 (체커보드 검출 시 끄는 게 좋음)
    # ---------------------------------------------------------------
    def set_emitter_enabled(self, enabled: bool) -> None:
        """IR 패턴 emitter on/off. 체커보드 같은 정밀 RGB 추출 시 끄는 게 유리.
        끄면 깊이 품질은 떨어지므로 캡처 후 다시 켜는 패턴을 권장."""
        if not self._connected or self.profile is None:
            return
        try:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if enabled else 0.0)
            logger.info(f"IR emitter {'ON' if enabled else 'OFF'}")
        except Exception as e:
            logger.warning(f"IR emitter 토글 실패: {e}")

    # ---------------------------------------------------------------
    # 캡처
    # ---------------------------------------------------------------
    def capture(self) -> Optional[RSFrame]:
        if not self._connected:
            logger.error("RealSense 미연결 — capture 불가")
            return None
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                return None
            # depth_scale 을 함께 묶어서 정적 메서드가 mm 변환할 수 있도록
            return {
                "depth": depth,
                "color": color,
                "depth_scale": self.depth_scale,
                "intrinsics": depth.profile.as_video_stream_profile().get_intrinsics(),
            }
        except Exception as e:
            logger.error(f"RealSense 캡처 실패: {e}")
            return None

    # ---------------------------------------------------------------
    # 프레임 → 표준 데이터 변환
    # ---------------------------------------------------------------
    @staticmethod
    def frame_to_2d_image(frame: RSFrame) -> Optional[np.ndarray]:
        try:
            color = frame["color"]
            return np.asanyarray(color.get_data()).copy()  # BGR (uint8)
        except Exception as e:
            logger.error(f"2D 이미지 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_point_cloud(frame: RSFrame) -> Optional[np.ndarray]:
        """
        depth → (H, W, 3) XYZ mm. 깊이 없는 픽셀은 NaN.
        depth-to-color 정렬됐으므로 color 픽셀(u,v)와 동일 인덱스의 XYZ 가 같은 점.
        """
        try:
            depth = frame["depth"]
            intr = frame["intrinsics"]
            depth_scale = frame["depth_scale"]

            depth_image = np.asanyarray(depth.get_data())  # (H, W) uint16 raw
            h, w = depth_image.shape

            # raw → meter → mm
            z_m = depth_image.astype(np.float32) * float(depth_scale)
            z_mm = z_m * 1000.0

            u = np.arange(w, dtype=np.float32)
            v = np.arange(h, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)

            # 핀홀 모델: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
            x = (uu - intr.ppx) * z_mm / intr.fx
            y = (vv - intr.ppy) * z_mm / intr.fy

            xyz = np.stack([x, y, z_mm], axis=-1)

            # 깊이가 0 인 픽셀(= 측정 실패)은 NaN 처리
            invalid = depth_image == 0
            xyz[invalid] = np.nan
            return xyz.astype(np.float32)
        except Exception as e:
            logger.error(f"포인트 클라우드 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_normals(frame: RSFrame) -> Optional[np.ndarray]:
        """
        RealSense 는 normal 직접 제공 안 함 → None.
        호출 측은 None 받으면 점군에서 자체 추정한다 (본 시스템 코드는
        `_orient_normals_outward`, `_fit_plane`, Open3D estimate_normals 등을
        이미 fallback 으로 보유).
        """
        return None

    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """현재 color stream 의 카메라 행렬 + 왜곡 (BaseCamera 표준 형식)."""
        if not self._connected or self.profile is None:
            logger.error("RealSense 미연결 — intrinsics 불가")
            return None
        try:
            color_profile = self.profile.get_stream(rs.stream.color)
            intr = color_profile.as_video_stream_profile().get_intrinsics()
            return {
                "camera_matrix": [
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                "dist_coeffs": list(intr.coeffs)[:5],  # k1, k2, p1, p2, k3
            }
        except Exception as e:
            logger.error(f"intrinsics 가져오기 실패: {e}")
            return None

    # ---------------------------------------------------------------
    # 저장
    # ---------------------------------------------------------------
    @staticmethod
    def save_point_cloud(frame: RSFrame, path: str) -> None:
        """
        RealSense 원본 포인트 클라우드를 PLY 로 저장.
        (Zivid 는 .zdf 가 네이티브지만 RealSense 는 .ply 가 흔함.)
        """
        try:
            pc = rs.pointcloud()
            color = frame["color"]
            depth = frame["depth"]
            pc.map_to(color)
            points = pc.calculate(depth)
            points.export_to_ply(path, color)
            logger.info(f"포인트 클라우드 저장 (PLY): {path}")
        except Exception as e:
            logger.error(f"포인트 클라우드 저장 실패: {e}")
