"""
Percipio Camera Module (FM815-IX-E1 등 vcamera SDK 호환 모델)
============================================================
BaseCamera 인터페이스 구현 — Zivid / RealSense 와 동일한 외부 API.

설계 결정:
  - Percipio vcamera SDK 는 **콜백 기반 비동기** 캡처 (FireSoftwareTrigger →
    FrameSetCallback). 본 시스템 BaseCamera 는 동기 capture() 모델이므로
    Queue 를 두고 콜백→Queue→capture() 동기 변환.
  - 콜백 안에서 frame_set 의 모든 데이터를 numpy 로 즉시 추출해 dict 로 저장.
    SDK 객체 수명이 콜백 밖에서 보장 안 될 가능성 회피.
  - 깊이 단위: depth.to_numpy() 는 raw int16, depth.scale_unit() 곱하면 mm.
    XYZ 점군은 핀홀 모델로 직접 계산 — `ImageProc.DepthImageToPointCloud()`
    결과의 단위가 SDK 버전마다 다를 수 있어 명시적으로 mm 통일.
  - depth → color 정합: SetMapDepthToTextureEnabled(True) 로 depth 를 color
    좌표계로 매핑 (RealSense `rs.align` 과 동일 컨셉). 이러면 color 픽셀
    (u,v) 와 동일 인덱스의 XYZ 가 같은 점.

알려진 한계 (FM815-IX-E1 기준):
  - 깊이 정밀도 1.5–1.8 mm Z @ 1 m (Zivid 의 ~25 배 거칠음).
    Hand-eye calibration 정확도 떨어지므로 PnP 방식 권장.
  - 표면 법선 직접 제공 없음 (SensorType.NormalMap 은 일부 상위 모델만)
    → frame_to_normals 는 None 반환, 호출 측이 자체 추정.
  - SDK 가 Linux Python 바인딩 wheel 형태 — `import vcamera as vcam`.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

from base_camera import BaseCamera

logger = logging.getLogger(__name__)

# Percipio vcamera SDK 는 별도 wheel/소스 빌드가 필요.
# import 실패 시 인스턴스화 시점에 명확히 에러를 알려준다.
try:
    import vcamera as vcam

    HAS_PERCIPIO = True
except Exception as e:
    vcam = None
    HAS_PERCIPIO = False
    _IMPORT_ERROR = str(e)


# Percipio frame 표현: 콜백에서 즉시 추출한 numpy 딕셔너리.
# (SDK frame_set 객체 수명 불확실 → 콜백 안에서 to_numpy().copy() 로 안전 복제)
PFrame = Dict[str, Any]


class PercipioCamera(BaseCamera):
    """
    Percipio vcamera SDK 카메라 (FM815-IX-E1 등). BaseCamera 구현체.

    캡처 흐름 (동기 변환):
      capture()
        ├─ self._frame_queue.queue.clear()    (예전 frame 버림)
        ├─ FireSoftwareTrigger()
        ├─ Queue.get(timeout=5)               (콜백이 frame_dict 채워넣을 때까지)
        └─ return frame_dict
    """

    # 콜백 동기화 타임아웃 (초)
    CAPTURE_TIMEOUT_SEC = 5.0

    def __init__(self):
        if not HAS_PERCIPIO:
            raise RuntimeError(
                f"vcamera (Percipio SDK) import 실패: {_IMPORT_ERROR}\n"
                "설치: /home/robotegra/Downloads/VcameraSDK-*/python/ 에서 "
                "build_wheel.py 또는 pip install . 로 vcamera wheel 빌드 후 설치."
            )

        # SDK 객체
        self._cam = None
        self._cam_info = None

        # 단일 인스턴스 capture() 호출 직렬화 + 콜백 frame 동기화용 queue
        self._frame_queue: "queue.Queue[PFrame]" = queue.Queue(maxsize=1)
        self._capture_lock = threading.Lock()

        # 상태
        self.settings: Optional[Dict[str, Any]] = None   # load_settings 후 path 기록
        self._connected = False
        self._capturing = False
        self._color_intrinsic: Optional[np.ndarray] = None
        self._color_distortion: Optional[list] = None
        self._depth_intrinsic: Optional[np.ndarray] = None

    # ===========================================================
    # 연결 / 설정
    # ===========================================================
    def connect(self) -> bool:
        try:
            vcam.CameraUtils.Init(True)
            infos = vcam.CameraUtils.DiscoverCameras()
            if not infos:
                logger.error("Percipio 카메라를 찾지 못했습니다 (DiscoverCameras 결과 비어 있음)")
                return False

            self._cam_info = infos[0]
            sn = getattr(self._cam_info, "serial_number", str(self._cam_info))
            logger.info(f"Percipio 카메라 발견: SN={sn}")

            self._cam = vcam.CameraFactory.GetCameraBySerialNumber(sn)
            status = self._cam.Connect()
            if not status:
                logger.error(f"Percipio Connect 실패: {status.message()}")
                self._cam = None
                return False

            # SingleFrame + SoftTrigger 모드 — capture() 호출마다 트리거를 한 번 발사
            if not self._set_feature_int("AcquisitionMode", 0):   # 0=SingleFrame
                self._cam.Disconnect()
                return False
            if not self._set_feature_int("TriggerSource", 8):     # 8=SoftTrigger
                self._cam.Disconnect()
                return False

            # Depth + Color 센서 활성화 (모델에 따라 한쪽만 있을 수도 있음)
            self._enable_sensor_if_present(vcam.SensorType.Depth, "Depth")
            self._enable_sensor_if_present(vcam.SensorType.Texture, "Color")

            # Depth 를 Color 좌표계로 매핑 (있다면). color 와 depth 가 같은 (u,v) 그리드.
            try:
                self._cam.SetMapDepthToTextureEnabled(True)
            except Exception as e:
                logger.warning(f"SetMapDepthToTextureEnabled 실패 (무시): {e}")

            # 왜곡 보정 활성화 (있다면) — 후속 핀홀 모델 계산이 정확해짐
            try:
                self._cam.SetUndistortionEnabled(True)
            except Exception as e:
                logger.warning(f"SetUndistortionEnabled 실패 (무시): {e}")

            # 콜백 등록 + 캡처 시작
            self._cam.RegisterFrameSetCallback(self._on_frame_set)
            if not self._cam.StartCapture():
                logger.error("Percipio StartCapture 실패")
                self._cam.Disconnect()
                self._cam = None
                return False
            self._capturing = True
            self._connected = True

            logger.info("Percipio 연결 성공 (SingleFrame + SoftTrigger 모드)")
            return True

        except Exception as e:
            logger.error(f"Percipio 연결 실패: {e}")
            self._cleanup_on_failure()
            return False

    def _enable_sensor_if_present(self, sensor_type, label: str) -> None:
        """존재하는 센서만 활성화. HasSensor 가 False 면 조용히 skip."""
        try:
            has, _ = self._cam.HasSensor(sensor_type)
        except Exception:
            has = True   # API 차이로 호출 실패 시, 무조건 활성화 시도
        if not has:
            logger.info(f"Percipio: {label} 센서 없음 — skip")
            return
        status = self._cam.SetSensorEnabled(sensor_type, True)
        if not status:
            logger.warning(f"Percipio {label} 센서 활성화 실패: {status.message()}")

    def _set_feature_int(self, name: str, value: int) -> bool:
        """Int 타입 feature 설정. 실패 시 False + 로그."""
        try:
            feat, status = self._cam.GetFeature(name)
            if not status:
                logger.error(f"GetFeature({name}) 실패: {status.message()}")
                return False
            status = feat.SetValue(value)
            if not status:
                logger.error(f"SetFeature({name}={value}) 실패: {status.message()}")
                return False
            return True
        except Exception as e:
            logger.error(f"feature {name}={value} 설정 예외: {e}")
            return False

    def _cleanup_on_failure(self) -> None:
        """connect 중간 실패 시 부분 자원 정리."""
        try:
            if self._cam is not None:
                if self._capturing:
                    self._cam.StopCapture()
                self._cam.Disconnect()
        except Exception:
            pass
        self._cam = None
        self._cam_info = None
        self._capturing = False
        self._connected = False

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._capturing:
                self._cam.StopCapture()
                self._capturing = False
        except Exception as e:
            logger.warning(f"Percipio StopCapture 중 오류 (무시): {e}")
        try:
            self._cam.Disconnect()
        except Exception as e:
            logger.warning(f"Percipio Disconnect 중 오류 (무시): {e}")
        self._cam = None
        self._cam_info = None
        self._connected = False
        logger.info("Percipio 연결 해제")

    def load_settings(self, path: str) -> bool:
        """
        Percipio 는 feature 묶음을 .fea / .ini / .json 파일로 저장/로드.
        Zivid YML 과 인터페이스 통일을 위해 같은 메서드명 사용.
        """
        if not self._connected or self._cam is None:
            logger.error("Percipio 미연결 — 설정 로드 불가")
            return False
        try:
            status = self._cam.LoadFeaturesFromFile(path)
            if not status:
                logger.error(f"LoadFeaturesFromFile 실패: {status.message()}")
                return False
            self.settings = {"_path": path}
            logger.info(f"Percipio 설정 로드 완료: {path}")
            return True
        except Exception as e:
            logger.error(f"Percipio 설정 로드 예외: {e}")
            return False

    # ===========================================================
    # 콜백 (비동기) → Queue (동기)
    # ===========================================================
    def _on_frame_set(self, frame_set) -> None:
        """SDK 콜백 — 받은 frame_set 의 데이터를 즉시 numpy 로 추출해 큐에 넣음."""
        try:
            entry: PFrame = {
                "color": None,
                "depth_mm": None,
                "xyz": None,
                "color_intrinsic": None,
                "color_distortion": None,
                "depth_intrinsic": None,
            }

            # ---- Color (Texture) ----
            try:
                color = frame_set.GetImage(vcam.SensorType.Texture)
            except Exception:
                color = None
            if color is not None and color.IsValid():
                color_np = color.to_numpy().copy()
                # OpenCV BGR 가정 — RGB 면 사용자가 시각화에서 색 반전 확인 후 cvtColor 추가
                entry["color"] = color_np
                try:
                    calib = color.calib_info()
                    if calib is not None:
                        entry["color_intrinsic"] = np.array(calib.intrinsic.matrix, dtype=np.float64)
                        coeffs = list(calib.distortion.coefficients)
                        entry["color_distortion"] = coeffs
                except Exception as e:
                    logger.debug(f"color calib 읽기 실패: {e}")

            # ---- Depth ----
            try:
                depth = frame_set.GetImage(vcam.SensorType.Depth)
            except Exception:
                depth = None
            if depth is not None and depth.IsValid():
                raw = depth.to_numpy().copy()
                # to_numpy() 는 raw int. scale_unit() 곱하면 mm.
                scale = 1.0
                try:
                    scale = float(depth.scale_unit())
                except Exception:
                    pass
                depth_mm = raw.astype(np.float32) * scale
                entry["depth_mm"] = depth_mm

                try:
                    calib = depth.calib_info()
                    if calib is not None:
                        entry["depth_intrinsic"] = np.array(calib.intrinsic.matrix, dtype=np.float64)
                except Exception as e:
                    logger.debug(f"depth calib 읽기 실패: {e}")

                # depth 가 color 좌표계로 매핑된 상태(SetMapDepthToTextureEnabled=True)
                # 라면 color intrinsic 으로 backproject. 미매핑이면 depth intrinsic 사용.
                intr_for_xyz = entry["color_intrinsic"] if entry["color_intrinsic"] is not None else entry["depth_intrinsic"]
                if intr_for_xyz is not None:
                    entry["xyz"] = self._backproject_to_xyz(depth_mm, intr_for_xyz)

            # 큐는 maxsize=1: 이전 frame 이 남아 있으면 버리고 새 것으로 교체
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(entry)

            # 인스턴스 캐시 (get_intrinsics 빠른 조회용)
            if entry["color_intrinsic"] is not None:
                self._color_intrinsic = entry["color_intrinsic"]
                self._color_distortion = entry["color_distortion"]
            if entry["depth_intrinsic"] is not None:
                self._depth_intrinsic = entry["depth_intrinsic"]

        except Exception as e:
            logger.error(f"Percipio 콜백 처리 실패: {e}")

    @staticmethod
    def _backproject_to_xyz(depth_mm: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        """
        depth(mm, H×W) + 3×3 intrinsic → (H, W, 3) XYZ mm. depth==0 인 픽셀은 NaN.
        핀홀 모델: X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=depth.
        """
        h, w = depth_mm.shape[:2]
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        z = depth_mm.astype(np.float32)

        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        xyz = np.stack([x, y, z], axis=-1)

        invalid = (depth_mm == 0) | np.isnan(depth_mm)
        xyz[invalid] = np.nan
        return xyz.astype(np.float32)

    # ===========================================================
    # 캡처 (동기)
    # ===========================================================
    def capture(self) -> Optional[PFrame]:
        if not self._connected or self._cam is None:
            logger.error("Percipio 미연결 — capture 불가")
            return None
        # 동시 호출 방지 (UI 가 빠르게 두 번 누르는 케이스)
        with self._capture_lock:
            # 이전에 남은 frame 이 있으면 버려서 fresh 한 것만 받기
            try:
                while True:
                    self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                status = self._cam.FireSoftwareTrigger()
                if not status:
                    logger.error(f"Percipio FireSoftwareTrigger 실패: {status.message()}")
                    return None
            except Exception as e:
                logger.error(f"Percipio trigger 예외: {e}")
                return None

            try:
                frame = self._frame_queue.get(timeout=self.CAPTURE_TIMEOUT_SEC)
            except queue.Empty:
                logger.error(f"Percipio 캡처 타임아웃 ({self.CAPTURE_TIMEOUT_SEC}s)")
                return None
            return frame

    # ===========================================================
    # 프레임 → 표준 데이터 변환 (정적 — frame dict 만 읽음)
    # ===========================================================
    @staticmethod
    def frame_to_2d_image(frame: PFrame) -> Optional[np.ndarray]:
        try:
            color = frame.get("color") if frame else None
            if color is None:
                return None
            return color.copy()   # (H, W, 3) BGR uint8
        except Exception as e:
            logger.error(f"2D 이미지 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_point_cloud(frame: PFrame) -> Optional[np.ndarray]:
        try:
            xyz = frame.get("xyz") if frame else None
            if xyz is None:
                return None
            return xyz.copy()     # (H, W, 3) float32 mm, invalid=NaN
        except Exception as e:
            logger.error(f"포인트 클라우드 추출 실패: {e}")
            return None

    @staticmethod
    def frame_to_normals(frame: PFrame) -> Optional[np.ndarray]:
        """Percipio FM 시리즈는 표면 normal 직접 제공 안 함 → None.
        호출 측은 점군에서 자체 추정 (본 시스템 코드는 fallback 보유)."""
        return None

    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """현재 color stream 의 카메라 행렬 + 왜곡 (BaseCamera 표준 형식).
        아직 한 프레임도 캡처 못 했으면 None 가능."""
        intr = self._color_intrinsic
        dist = self._color_distortion
        if intr is None:
            # 콜백을 한 번도 못 받았으면, 강제로 한 장 캡처해서 캐시 채움
            f = self.capture()
            if f is None:
                logger.error("intrinsics 조회용 캡처 실패")
                return None
            intr = self._color_intrinsic
            dist = self._color_distortion
        if intr is None:
            return None
        return {
            "camera_matrix": intr.tolist(),
            # OpenCV 표준은 5/8/14 길이. Percipio 가 12개 줄 수도 있어 최대 14 까지 자름.
            "dist_coeffs": list(dist)[:14] if dist else [0.0] * 5,
        }

    # ===========================================================
    # 저장
    # ===========================================================
    @staticmethod
    def save_point_cloud(frame: PFrame, path: str) -> None:
        """
        XYZ + color 를 ASCII PLY 로 저장 (RealSense 와 동일 포맷).
        의존성 추가 없이 자체 writer.
        """
        try:
            xyz = frame.get("xyz") if frame else None
            color = frame.get("color") if frame else None
            if xyz is None:
                logger.error("xyz 없음 — 포인트 클라우드 저장 불가")
                return

            pts = xyz.reshape(-1, 3)
            valid = ~np.any(np.isnan(pts), axis=1)
            pts = pts[valid]

            colors = None
            if color is not None and color.shape[:2] == xyz.shape[:2]:
                # BGR → RGB (PLY 표준은 RGB)
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).reshape(-1, 3)[valid]
                colors = rgb.astype(np.uint8)

            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(pts)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                if colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                f.write("end_header\n")
                if colors is not None:
                    for p, c in zip(pts, colors):
                        f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
                else:
                    for p in pts:
                        f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
            logger.info(f"포인트 클라우드 저장 (PLY): {path}, {len(pts)} pts")
        except Exception as e:
            logger.error(f"포인트 클라우드 저장 실패: {e}")
