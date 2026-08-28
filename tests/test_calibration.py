"""calibration.py 순수 함수 테스트 — 로봇/카메라 없이 수학만 검증한다.

핵심 성질:
- tcp ↔ 4×4 동차변환이 왕복(round-trip)에서 값이 보존되는가
- CalibrationContext 가 양 모드에서 올바른 변환/역변환을 하는가
- 손상된 입력에서 조용히 이상한 좌표를 내놓지 않고 안전하게 실패하는가
"""

import json

import numpy as np
import pytest

from calibration import (
    CalibrationContext,
    tcp_to_homogeneous,
    homogeneous_to_tcp,
    compute_approach_pose,
)


class TestTcpHomogeneous:
    def test_roundtrip(self):
        """tcp → 4×4 → tcp 왕복에서 위치/각도가 보존돼야 한다."""
        tcp = {"x": 123.4, "y": -56.7, "z": 890.1, "a": 45.0, "b": -30.0, "c": 170.0}
        back = homogeneous_to_tcp(tcp_to_homogeneous(tcp))
        for k in tcp:
            assert back[k] == pytest.approx(tcp[k], abs=1e-6), k

    def test_identity(self):
        """0 자세는 단위 행렬."""
        T = tcp_to_homogeneous({"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 0})
        assert np.allclose(T, np.eye(4))

    def test_translation_only(self):
        T = tcp_to_homogeneous({"x": 10, "y": 20, "z": 30, "a": 0, "b": 0, "c": 0})
        assert np.allclose(T[:3, 3], [10, 20, 30])
        assert np.allclose(T[:3, :3], np.eye(3))

    def test_rotation_is_orthonormal(self):
        T = tcp_to_homogeneous({"x": 0, "y": 0, "z": 0, "a": 33.0, "b": -71.0, "c": 155.0})
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_abc_wraparound_equivalence(self):
        """A=+190° 와 A=-170° 는 같은 회전 행렬이어야 한다 (호스 꼬임 이슈의 근원)."""
        T1 = tcp_to_homogeneous({"x": 0, "y": 0, "z": 0, "a": 190.0, "b": 0, "c": 0})
        T2 = tcp_to_homogeneous({"x": 0, "y": 0, "z": 0, "a": -170.0, "b": 0, "c": 0})
        assert np.allclose(T1, T2, atol=1e-9)


class TestApproachPose:
    def test_tool_z_aligns_to_negative_normal(self):
        """수직 접근 자세: Tool +Z 축이 표면 법선의 반대 방향이어야 한다."""
        target = np.array([100.0, 200.0, 50.0])
        normal = np.array([0.0, 0.0, 1.0])  # 수평 표면, 법선 위쪽
        cur = {"x": 0, "y": 0, "z": 300, "a": 0, "b": 0, "c": 180}
        pose = compute_approach_pose(target, normal, cur)
        assert (pose["x"], pose["y"], pose["z"]) == pytest.approx(tuple(target))
        R = tcp_to_homogeneous(pose)[:3, :3]
        assert np.allclose(R[:, 2], -normal, atol=1e-6)  # Tool +Z = -법선

    def test_tilted_surface(self):
        """기울어진 표면에서도 Tool +Z = -법선 이 성립해야 한다."""
        normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        pose = compute_approach_pose(np.zeros(3), normal, {"x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 180})
        R = tcp_to_homogeneous(pose)[:3, :3]
        assert np.allclose(R[:, 2], -normal, atol=1e-6)


class TestCalibrationContext:
    def _write(self, tmp_path, T, mode="eye_to_hand"):
        p = tmp_path / "calib.json"
        p.write_text(json.dumps({"transformation_matrix": T.tolist(), "mode": mode}))
        return str(p)

    def test_unloaded_returns_none(self):
        c = CalibrationContext()
        assert not c.loaded
        assert c.cam_to_base([1, 2, 3]) == (None, None)
        assert c.T_cam_to_base() is None

    def test_load_and_transform_eye_to_hand(self, tmp_path):
        T = np.eye(4)
        T[:3, 3] = [100.0, 0.0, 0.0]
        c = CalibrationContext()
        name = c.load(self._write(tmp_path, T))
        assert name == "calib.json" and c.mode == "eye_to_hand" and c.loaded
        pb, nb = c.cam_to_base([1, 2, 3], [0, 0, 1])
        assert np.allclose(pb, [101, 2, 3])
        assert np.allclose(nb, [0, 0, 1])  # 순수 병진이라 법선 불변

    def test_points_roundtrip(self, tmp_path):
        """cam→base→cam 왕복은 원본을 복원해야 한다 (회전 포함)."""
        tcp = {"x": 5, "y": 6, "z": 7, "a": 30, "b": 10, "c": -20}
        T = tcp_to_homogeneous(tcp)
        c = CalibrationContext()
        c.load(self._write(tmp_path, T))
        pts = np.array([[1.0, 2, 3], [40, 50, 60]])
        back = c.base_to_cam_points(c.cam_to_base_points(pts))
        assert np.allclose(back, pts, atol=1e-9)

    def test_eye_in_hand_requires_tcp(self, tmp_path):
        c = CalibrationContext()
        c.load(self._write(tmp_path, np.eye(4), mode="eye_in_hand"))
        assert c.cam_to_base([1, 2, 3]) == (None, None)  # TCP 없으면 안전 실패
        tcp = {"x": 100, "y": 0, "z": 0, "a": 0, "b": 0, "c": 0}
        pb, _ = c.cam_to_base([1, 2, 3], tcp=tcp)
        assert np.allclose(pb, [101, 2, 3])

    def test_bad_matrix_shape_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"transformation_matrix": [[1, 2], [3, 4]]}))
        c = CalibrationContext()
        with pytest.raises(Exception):
            c.load(str(p))
        assert not c.loaded  # 실패 후에도 미로드 상태 유지
