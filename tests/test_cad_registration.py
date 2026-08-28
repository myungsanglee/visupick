"""cad_registration.py 테스트 — 합성 장면으로 정합/자세 변환 검증.

멀티 인스턴스 정합(느린 편, ~수 초)과 자세→TCP 변환의 수학적 성질을 확인한다.
"""

import numpy as np
import open3d as o3d
import pytest

import cad_registration as cr


def box_model(w=30.0, d=20.0, h=10.0, n=2000):
    mesh = o3d.geometry.TriangleMesh.create_box(w, d, h)
    return mesh.sample_points_uniformly(n)


class TestMultiInstance:
    def test_two_instances_found(self):
        model = box_model()
        pts = np.asarray(model.points)
        scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pts, pts + [100.0, 50.0, 0.0]])))
        res, log = cr.cad_match_multi_instance(scene, model, voxel_size=3.0, max_instances=3, fitness_threshold=0.5)
        assert len(res) == 2
        for r in res:
            assert r["fitness"] > 0.9
        # 두 인스턴스의 위치 차이가 심어 둔 오프셋과 일치해야 한다
        t = sorted(r["transformation"][:3, 3].tolist() for r in res)
        diff = np.array(t[1]) - np.array(t[0])
        assert np.allclose(diff, [100.0, 50.0, 0.0], atol=3.0)

    def test_empty_scene(self):
        model = box_model()
        scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(200, 3) * 5.0))
        res, log = cr.cad_match_multi_instance(scene, model, voxel_size=3.0, max_instances=2, fitness_threshold=0.9)
        assert len(res) == 0


class TestObjectPoseToTcp:
    def test_identity_calib_position(self):
        """항등 캘리브레이션이면 TCP 위치 = 객체 위치."""
        T = np.eye(4)
        T[:3, 3] = [10.0, 20.0, 30.0]
        tcp = cr.object_pose_to_tcp(T, np.eye(4), "eye_to_hand", None)
        assert (tcp["x"], tcp["y"], tcp["z"]) == pytest.approx((10.0, 20.0, 30.0))

    def test_calib_translation_applied(self):
        T_obj = np.eye(4)
        T_calib = np.eye(4)
        T_calib[:3, 3] = [500.0, 0.0, 0.0]
        tcp = cr.object_pose_to_tcp(T_obj, T_calib, "eye_to_hand", None)
        assert tcp["x"] == pytest.approx(500.0)

    def test_grasp_offset_in_object_frame(self):
        """grasp offset 은 객체 좌표계 기준으로 적용돼야 한다."""
        T_obj = np.eye(4)
        tcp0 = cr.object_pose_to_tcp(T_obj, np.eye(4), "eye_to_hand", None)
        tcp1 = cr.object_pose_to_tcp(T_obj, np.eye(4), "eye_to_hand", None, grasp_offset_xyz=(5.0, 0.0, 0.0))
        assert abs((tcp1["x"] - tcp0["x"])) == pytest.approx(5.0, abs=1e-6)


class TestSuggestRotation:
    def test_flat_surface_tool_down(self):
        """수평 표면(법선 +Z)이면 Tool +Z 가 아래를 봐야 한다 (C≈180)."""
        abc = cr.suggest_rotation_from_normal(np.array([0.0, 0.0, 1.0]), "Z", False)
        assert abc is not None
        from calibration import tcp_to_homogeneous

        R = tcp_to_homogeneous({"x": 0, "y": 0, "z": 0, "a": abc[0], "b": abc[1], "c": abc[2]})[:3, :3]
        assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-6)
