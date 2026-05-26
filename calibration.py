"""
Hand-Eye Calibration Module
Eye-to-Hand / Eye-in-Hand calibration 계산
Zivid 3D 포인트 클라우드 기반 체커보드 포즈 추정
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


def tcp_to_homogeneous(tcp: Dict[str, float]) -> np.ndarray:
    """
    KUKA TCP (x, y, z, a, b, c) → 4x4 동차 변환 행렬

    KUKA 오일러 각도 규칙: Z-Y-X (A=Rz, B=Ry, C=Rx)

    Args:
        tcp: {"x", "y", "z", "a", "b", "c"} (mm, 도)

    Returns:
        4x4 변환 행렬
    """
    t = np.array([tcp["x"], tcp["y"], tcp["z"]])

    # KUKA: A=Rz, B=Ry, C=Rx (ZYX 순서, 내재 회전)
    a_rad = np.radians(tcp["a"])
    b_rad = np.radians(tcp["b"])
    c_rad = np.radians(tcp["c"])

    rot = Rotation.from_euler("ZYX", [a_rad, b_rad, c_rad])
    R = rot.as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def homogeneous_to_tcp(T: np.ndarray) -> Dict[str, float]:
    """
    4x4 동차 변환 행렬 → KUKA TCP (x, y, z, a, b, c)

    Returns:
        {"x", "y", "z", "a", "b", "c"} (mm, 도)
    """
    t = T[:3, 3]
    R = T[:3, :3]

    rot = Rotation.from_matrix(R)
    a_rad, b_rad, c_rad = rot.as_euler("ZYX")

    return {
        "x": float(t[0]),
        "y": float(t[1]),
        "z": float(t[2]),
        "a": float(np.degrees(a_rad)),
        "b": float(np.degrees(b_rad)),
        "c": float(np.degrees(c_rad)),
    }


def _sample_3d_at_pixel(xyz: np.ndarray, px: int, py: int, patch: int = 3) -> Optional[np.ndarray]:
    """
    픽셀 주변 패치의 유효한 3D 포인트 평균을 반환 (노이즈 감소)

    Args:
        xyz: 포인트 클라우드 (H, W, 3)
        px, py: 픽셀 좌표
        patch: 패치 반경 (3이면 7x7 영역)
    """
    h, w = xyz.shape[:2]
    x0, x1 = max(0, px - patch), min(w, px + patch + 1)
    y0, y1 = max(0, py - patch), min(h, py + patch + 1)

    region = xyz[y0:y1, x0:x1].reshape(-1, 3)
    valid = region[~np.any(np.isnan(region), axis=1)]
    if len(valid) == 0:
        return None
    return valid.mean(axis=0)


def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D 포인트들에 최적 평면 피팅 (SVD)

    Returns:
        (centroid, normal) - 평면의 중심점과 법선 벡터
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # 가장 작은 특이값에 해당하는 벡터 = 법선
    return centroid, normal


def _project_onto_plane(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """포인트들을 평면에 수직 투영"""
    diff = points - centroid
    dist = diff @ normal
    return points - dist[:, None] * normal[None, :]


def estimate_pose_from_pointcloud(
    corners_2d: np.ndarray,
    xyz: np.ndarray,
    board_size: Tuple[int, int],
    square_size: float,
    patch_radius: int = 3,
    use_plane_fitting: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    2D 체커보드 코너 좌표 + 3D 포인트 클라우드 → 체커보드-카메라 변환

    개선점:
    - 각 코너의 3D 좌표를 주변 패치 평균으로 구해 노이즈 감소
    - 모든 유효 코너로 평면 피팅 후 각 포인트를 평면에 투영 (체커보드가 평면이라는 제약 활용)

    Args:
        corners_2d: 체커보드 코너 픽셀 좌표 (N, 1, 2)
        xyz: 포인트 클라우드 (H, W, 3) mm
        board_size: 체커보드 내부 코너 수 (가로, 세로)
        square_size: 체커보드 한 칸 크기 (mm)
        patch_radius: 픽셀 주변 평균 영역 반경
        use_plane_fitting: 체커보드 평면 피팅 사용 여부

    Returns:
        (R, t, rmse) 또는 None
    """
    # 체커보드 물리 좌표
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    h, w = xyz.shape[:2]
    cam_points = []
    obj_points = []

    for i, corner in enumerate(corners_2d.reshape(-1, 2)):
        px, py = int(round(corner[0])), int(round(corner[1]))
        if not (0 <= px < w and 0 <= py < h):
            continue

        # 패치 평균으로 3D 포인트 추출 (노이즈 감소)
        point_3d = _sample_3d_at_pixel(xyz, px, py, patch=patch_radius)
        if point_3d is None:
            continue

        cam_points.append(point_3d)
        obj_points.append(objp[i])

    if len(cam_points) < 4:
        logger.error(f"유효한 3D 포인트가 부족합니다 ({len(cam_points)}/{len(corners_2d)})")
        return None

    cam_points = np.array(cam_points, dtype=np.float64)
    obj_points = np.array(obj_points, dtype=np.float64)

    # 체커보드는 평면이라는 제약 활용: 평면 피팅 후 투영
    if use_plane_fitting and len(cam_points) >= 4:
        centroid, normal = _fit_plane(cam_points)
        cam_points_projected = _project_onto_plane(cam_points, centroid, normal)
        plane_residuals = np.linalg.norm(cam_points - cam_points_projected, axis=1)
        logger.info(
            f"평면 피팅 - 포인트 수: {len(cam_points)}, "
            f"평면 잔차 평균: {plane_residuals.mean():.3f}mm, 최대: {plane_residuals.max():.3f}mm"
        )
        cam_points = cam_points_projected

    # Rigid transform (SVD)
    obj_centroid = obj_points.mean(axis=0)
    cam_centroid = cam_points.mean(axis=0)
    obj_centered = obj_points - obj_centroid
    cam_centered = cam_points - cam_centroid

    H = obj_centered.T @ cam_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cam_centroid - R @ obj_centroid

    # RMSE 계산
    transformed = (R @ obj_points.T).T + t
    errors = np.linalg.norm(transformed - cam_points, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    logger.info(f"Rigid transform RMSE: {rmse:.4f} mm (max: {errors.max():.4f} mm)")

    return R, t.reshape(3, 1), rmse


def compute_hand_eye(
    data_dir: str,
    board_size: Tuple[int, int] = (8, 6),
    square_size: float = 25.0,
    mode: str = "eye_to_hand",
    pose_method: str = "auto",
    return_metric: bool = False,
) -> Optional[np.ndarray]:
    """
    저장된 데이터로 hand-eye calibration 수행.

    Args:
        data_dir: 데이터 디렉토리 (pose_001, pose_002, ... 포함)
        board_size: 체커보드 내부 코너 수 (가로, 세로)
        square_size: 체커보드 한 칸 크기 (mm)
        mode: "eye_to_hand" 또는 "eye_in_hand"
        pose_method: 체커보드 자세 추정 방식.
            "auto"       — 포인트 클라우드 있으면 3D 직접 매칭(SVD), 없으면 solvePnP (기본)
            "pointcloud" — 강제로 3D 직접 매칭만 사용 (포인트 클라우드 없으면 그 포즈 스킵)
            "pnp"        — 강제로 solvePnP 만 사용 (저정밀 깊이 카메라용. RealSense 등)
        호출자가 두 방법으로 각각 호출해 일관성 metric 비교 가능 ("compare" 모드).

    Returns:
        4x4 변환 행렬 (카메라↔로봇 관계) 또는 None
    """
    data_path = Path(data_dir)
    pose_dirs = sorted(data_path.glob("pose_*"))

    if len(pose_dirs) < 3:
        logger.error(f"최소 3개의 포즈가 필요합니다 (현재: {len(pose_dirs)})")
        return None

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    successful_poses = 0

    for pose_dir in pose_dirs:
        tcp_file = pose_dir / "tcp.json"
        image_file = pose_dir / "image.png"
        xyz_file = pose_dir / "pointcloud_xyz.npy"

        if not tcp_file.exists() or not image_file.exists():
            logger.warning(f"데이터 누락: {pose_dir.name}")
            continue

        # TCP → 변환 행렬
        with open(tcp_file) as f:
            tcp_data = json.load(f)
        T_gripper2base = tcp_to_homogeneous(tcp_data)

        # 이미지에서 체커보드 검출
        image = cv2.imread(str(image_file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            logger.warning(f"체커보드 검출 실패: {pose_dir.name}")
            continue

        # 서브픽셀 보정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 체커보드 자세 추정 — pose_method 에 따라 분기
        R_target2cam, t_target2cam = None, None

        use_pointcloud = pose_method in ("auto", "pointcloud") and xyz_file.exists()
        use_pnp_only = pose_method == "pnp"
        pnp_fallback = pose_method == "auto" and not xyz_file.exists()

        if use_pointcloud:
            xyz = np.load(str(xyz_file))
            result = estimate_pose_from_pointcloud(corners, xyz, board_size, square_size)
            if result is None:
                if pose_method == "pointcloud":
                    logger.warning(f"3D 포즈 추정 실패 ('pointcloud' 강제 모드): {pose_dir.name}")
                    continue
                # auto 모드: PnP fallback 으로 폴백
                pnp_fallback = True
            else:
                R_target2cam, t_target2cam, _rmse = result

        if use_pnp_only or pnp_fallback:
            objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
            objp *= square_size

            intrinsics_file = data_path / "intrinsics.json"
            if not intrinsics_file.exists():
                logger.error("intrinsics.json 없음, solvePnP 불가")
                continue
            with open(intrinsics_file) as f:
                intr = json.load(f)
            camera_matrix = np.array(intr["camera_matrix"])
            dist_coeffs = np.array(intr["dist_coeffs"])

            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if not success:
                logger.warning(f"solvePnP 실패: {pose_dir.name}")
                continue
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.reshape(3, 1)

        if R_target2cam is None:
            logger.warning(f"포즈 추정 결과 없음, 스킵: {pose_dir.name}")
            continue

        R_gripper2base_list.append(T_gripper2base[:3, :3])
        t_gripper2base_list.append(T_gripper2base[:3, 3].reshape(3, 1))
        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(t_target2cam)

        successful_poses += 1
        logger.info(f"포즈 로드 성공: {pose_dir.name}")

    logger.info(f"유효 포즈: {successful_poses}/{len(pose_dirs)}")

    if successful_poses < 3:
        logger.error(f"유효 포즈가 3개 미만입니다 ({successful_poses}개)")
        return None

    # Eye-to-Hand이면 gripper→base 역행렬 사용
    if mode == "eye_to_hand":
        R_in_list = []
        t_in_list = []
        for R_g, t_g in zip(R_gripper2base_list, t_gripper2base_list):
            R_in_list.append(R_g.T)
            t_in_list.append(-R_g.T @ t_g)
    elif mode == "eye_in_hand":
        R_in_list = R_gripper2base_list
        t_in_list = t_gripper2base_list
    else:
        logger.error(f"알 수 없는 모드: {mode}")
        return None

    # 1차: 모든 포즈 사용
    pose_names = [p.name for p in pose_dirs if (p / "tcp.json").exists() and (p / "image.png").exists()]
    keep_idx = list(range(len(R_in_list)))

    # Greedy outlier 제거: 매번 최대 잔차 포즈를 제거하고 metric이 개선되는 한 반복
    min_poses = max(8, len(keep_idx) // 2)  # 최소 8개 또는 원본의 절반까지만 제거 허용

    best_T = None
    best_metric = float("inf")
    best_name = None
    best_keep = list(keep_idx)

    iteration = 0
    while True:
        iteration += 1
        R_in_sub = [R_in_list[i] for i in keep_idx]
        t_in_sub = [t_in_list[i] for i in keep_idx]
        R_tc_sub = [R_target2cam_list[i] for i in keep_idx]
        t_tc_sub = [t_target2cam_list[i] for i in keep_idx]

        logger.info(f"=== Iteration {iteration} (포즈 {len(keep_idx)}개) ===")
        cand_T, cand_name, _ = _try_all_methods(
            R_in_sub, t_in_sub, R_tc_sub, t_tc_sub, mode,
        )
        if cand_T is None:
            logger.error("모든 알고리즘이 실패했습니다")
            return None

        refined_T = _refine_hand_eye(cand_T, R_in_sub, t_in_sub, R_tc_sub, t_tc_sub, mode)
        refined_metric, refined_per_pose = _evaluate_hand_eye(
            refined_T, R_in_sub, t_in_sub, R_tc_sub, t_tc_sub, mode,
        )
        logger.info(
            f"[{cand_name}+NLO] 일관성 오차: mean={refined_metric:.3f}mm, "
            f"max={max(refined_per_pose):.3f}mm"
        )

        if refined_metric < best_metric:
            best_T = refined_T
            best_metric = refined_metric
            best_name = f"{cand_name}+NLO"
            best_keep = list(keep_idx)
        else:
            # 더 이상 개선되지 않으면 종료
            logger.info(f"개선되지 않음 ({refined_metric:.3f} ≥ {best_metric:.3f}). 중단.")
            break

        if len(keep_idx) - 1 < min_poses:
            logger.info(f"최소 포즈 수 {min_poses}에 도달. 중단.")
            break

        # 가장 큰 잔차 포즈 제거 시도
        per_arr = np.array(refined_per_pose)
        max_idx = int(np.argmax(per_arr))
        removed_name = pose_names[keep_idx[max_idx]]
        logger.info(f"다음 시도: {removed_name} 제거 (잔차 {per_arr[max_idx]:.3f}mm)")
        keep_idx.pop(max_idx)

    keep_idx = best_keep

    # 최종 포즈별 잔차 로그
    R_in_final = [R_in_list[i] for i in keep_idx]
    t_in_final = [t_in_list[i] for i in keep_idx]
    R_tc_final = [R_target2cam_list[i] for i in keep_idx]
    t_tc_final = [t_target2cam_list[i] for i in keep_idx]
    _, per_pose = _evaluate_hand_eye(best_T, R_in_final, t_in_final, R_tc_final, t_tc_final, mode)
    final_names = [pose_names[i] for i in keep_idx]
    sorted_idx = np.argsort(per_pose)[::-1]
    logger.info("포즈별 잔차 (큰 순서, 상위 5개):")
    for idx in sorted_idx[:5]:
        logger.info(f"  {final_names[idx]}: {per_pose[idx]:.3f} mm")

    logger.info(f"=> 최종 ({best_name}): mean={best_metric:.3f}mm, 사용 포즈: {len(keep_idx)}/{len(R_in_list)}")
    logger.info(f"변환 행렬:\n{best_T}")
    if return_metric:
        return {
            "T": best_T,
            "metric_mean": float(best_metric),
            "n_used": len(keep_idx),
            "n_total": len(R_in_list),
            "algorithm": best_name,
        }
    return best_T


def _try_all_methods(
    R_in_list, t_in_list, R_target2cam_list, t_target2cam_list, mode: str,
) -> Tuple[Optional[np.ndarray], Optional[str], float]:
    """여러 hand-eye 알고리즘을 시도하고 가장 일관성 좋은 결과 반환"""
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    best_T = None
    best_metric = float("inf")
    best_name = None
    for name, method in methods.items():
        try:
            R_out, t_out = cv2.calibrateHandEye(
                R_in_list, t_in_list, R_target2cam_list, t_target2cam_list, method=method,
            )
            T = np.eye(4)
            T[:3, :3] = R_out
            T[:3, 3] = t_out.flatten()

            metric, per_pose = _evaluate_hand_eye(
                T, R_in_list, t_in_list, R_target2cam_list, t_target2cam_list, mode,
            )
            logger.info(f"  [{name}] mean={metric:.3f}mm, max={max(per_pose):.3f}mm")
            if metric < best_metric:
                best_metric = metric
                best_T = T
                best_name = name
        except Exception as e:
            logger.warning(f"  [{name}] 실패: {e}")
    return best_T, best_name, best_metric


def estimate_normal_at_pixel(
    xyz: np.ndarray,
    px: int,
    py: int,
    patch_radius: int = 15,
    normals: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    특정 픽셀 주변의 3D 포인트로 국소 평면을 피팅하여 법선 벡터 반환

    Zivid per-pixel normals는 미분 기반이라 노이즈가 큼.
    XYZ 포인트로 평면 피팅하면 많은 점이 평균화되어 각도 정밀도가 더 높음.

    Args:
        xyz: 포인트 클라우드 (H, W, 3)
        px, py: 타겟 픽셀
        patch_radius: 주변 영역 반경 (15면 31x31 영역, 약 900점)
        normals: Zivid 법선 맵 (선택) - 평면 피팅 실패시 fallback용

    Returns:
        단위 법선 벡터 (3,) - 카메라를 향하도록 정규화됨, 또는 None
    """
    h, w = xyz.shape[:2]
    if not (0 <= px < w and 0 <= py < h):
        return None

    x0, x1 = max(0, px - patch_radius), min(w, px + patch_radius + 1)
    y0, y1 = max(0, py - patch_radius), min(h, py + patch_radius + 1)

    region = xyz[y0:y1, x0:x1].reshape(-1, 3)
    valid = region[~np.any(np.isnan(region), axis=1)]

    if len(valid) >= 10:
        # 국소 XYZ 평면 피팅 (주 방식)
        _, normal = _fit_plane(valid)

        # 품질 체크: 평면에서 벗어난 점이 많으면 fallback
        centroid = valid.mean(axis=0)
        residuals = np.abs((valid - centroid) @ normal)
        rmse = np.sqrt(np.mean(residuals ** 2))
        logger.info(
            f"국소 평면 피팅: {len(valid)}점, RMSE={rmse:.3f}mm "
            f"(patch {2 * patch_radius + 1}x{2 * patch_radius + 1})"
        )

        if np.linalg.norm(normal) >= 1e-6:
            normal = normal / np.linalg.norm(normal)
            if normal[2] > 0:
                normal = -normal
            return normal

    # Fallback: Zivid per-pixel normals 평균
    if normals is not None:
        region_n = normals[y0:y1, x0:x1].reshape(-1, 3)
        valid_n = region_n[~np.any(np.isnan(region_n), axis=1)]
        if len(valid_n) >= 3:
            mean_normal = valid_n.mean(axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm >= 1e-6:
                mean_normal = mean_normal / norm
                if mean_normal[2] > 0:
                    mean_normal = -mean_normal
                logger.info("XYZ 평면 피팅 실패 → Zivid normals 사용")
                return mean_normal

    return None


def compute_approach_pose(
    target_base: np.ndarray,
    normal_base: np.ndarray,
    current_tcp: Dict[str, float],
) -> Dict[str, float]:
    """
    타겟 위치에 법선 방향으로 수직 접근하는 로봇 TCP 자세 계산

    - Tool의 -Z축을 법선 방향(표면 바깥쪽)과 정렬 → Tool이 표면을 향해 찌르는 자세
    - X/Y 축은 현재 TCP의 X축을 평면에 투영하여 결정 (자세 변화 최소화)

    Args:
        target_base: 타겟 위치 (로봇 base 좌표계), shape (3,)
        normal_base: 표면 법선 벡터 (로봇 base 좌표계), shape (3,)
        current_tcp: 현재 TCP (X축 방향 참고용)

    Returns:
        {"x", "y", "z", "a", "b", "c"} - 수직 접근 자세
    """
    normal_base = normal_base / np.linalg.norm(normal_base)

    # Tool +Z는 "작업 방향" (보통 툴팁이 향하는 쪽)
    # 표면에 수직 접근하려면 Tool +Z가 표면을 향해야 함 = 법선의 반대 방향
    new_z = -normal_base

    # 현재 TCP의 회전 행렬에서 X축 추출
    cur_T = tcp_to_homogeneous(current_tcp)
    cur_R = cur_T[:3, :3]
    cur_x = cur_R[:, 0]

    # 현재 X축을 new_z에 수직인 평면으로 투영
    new_x = cur_x - np.dot(cur_x, new_z) * new_z
    nx_norm = np.linalg.norm(new_x)
    if nx_norm < 1e-6:
        # 현재 X축이 new_z와 거의 평행인 경우 → Y축을 대안으로 사용
        cur_y = cur_R[:, 1]
        new_x = cur_y - np.dot(cur_y, new_z) * new_z
        nx_norm = np.linalg.norm(new_x)
        if nx_norm < 1e-6:
            # 둘 다 실패하면 월드 X축 기본값
            new_x = np.array([1.0, 0.0, 0.0])
            new_x = new_x - np.dot(new_x, new_z) * new_z
            nx_norm = np.linalg.norm(new_x)

    new_x = new_x / nx_norm
    new_y = np.cross(new_z, new_x)

    R_new = np.column_stack([new_x, new_y, new_z])

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = target_base

    result = homogeneous_to_tcp(T_new)

    # ABC unwrap: KUKA Euler ZYX 분해 결과가 ±180° wrap 경계에서 튀어
    # 현재 TCP와 "같은 자세인데 다른 표현"이 되어 IK가 큰 회전을 선택하는 것을 방지.
    # 각 각도를 현재 TCP에 가장 가까운 모듈로-360 표현으로 조정.
    for axis in ("a", "b", "c"):
        if axis in current_tcp:
            ref = float(current_tcp[axis])
            diff = (result[axis] - ref + 180.0) % 360.0 - 180.0
            result[axis] = ref + diff
    return result


def _pack_T(T: np.ndarray) -> np.ndarray:
    """4x4 변환 행렬 → 6-vector (rotvec + translation)"""
    rvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([rvec, T[:3, 3]])


def _unpack_T(params: np.ndarray) -> np.ndarray:
    """6-vector → 4x4 변환 행렬"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:]
    return T


def _refine_hand_eye(
    T_init: np.ndarray,
    R_in_list, t_in_list,
    R_target2cam_list, t_target2cam_list,
    mode: str,
) -> np.ndarray:
    """
    Non-linear optimization으로 hand-eye 변환 정밀화

    목적: 모든 포즈에서 target의 ref 좌표계 위치 분산 최소화
    (Eye-to-Hand: target은 gripper에 고정 → target_in_gripper가 일정해야 함)
    (Eye-in-Hand: target은 월드에 고정 → target_in_base가 일정해야 함)
    """
    T_in_arr = np.array([
        np.block([[R_in, t_in.reshape(3, 1)], [np.zeros(3), 1]])
        for R_in, t_in in zip(R_in_list, t_in_list)
    ])
    T_tc_arr = np.array([
        np.block([[R_tc, t_tc.reshape(3, 1)], [np.zeros(3), 1]])
        for R_tc, t_tc in zip(R_target2cam_list, t_target2cam_list)
    ])

    def residuals(params):
        T_est = _unpack_T(params)
        # T_target_in_ref = T_in @ T_est @ T_tc (Eye-to-Hand / Eye-in-Hand 모두 동일 형태)
        T_chain = T_in_arr @ T_est @ T_tc_arr
        positions = T_chain[:, :3, 3]  # (N, 3)
        mean_pos = positions.mean(axis=0)
        return (positions - mean_pos).flatten()

    try:
        result = least_squares(
            residuals, _pack_T(T_init), method="lm", max_nfev=500,
        )
        T_refined = _unpack_T(result.x)

        # 개선 확인
        init_metric, _ = _evaluate_hand_eye(T_init, R_in_list, t_in_list, R_target2cam_list, t_target2cam_list, mode)
        refined_metric, _ = _evaluate_hand_eye(T_refined, R_in_list, t_in_list, R_target2cam_list, t_target2cam_list, mode)
        logger.info(f"  Non-linear refine: {init_metric:.3f}mm → {refined_metric:.3f}mm")
        return T_refined if refined_metric < init_metric else T_init
    except Exception as e:
        logger.warning(f"Non-linear refinement 실패: {e}")
        return T_init


def _evaluate_hand_eye(
    T_result: np.ndarray,
    R_in_list, t_in_list,
    R_target2cam_list, t_target2cam_list,
    mode: str,
) -> Tuple[float, list]:
    """
    Hand-eye 결과의 일관성 평가

    Eye-to-Hand: target(체커보드)이 로봇 플랜지에 고정 → 모든 포즈에서 target의 base 좌표가 동일해야 함
    Eye-in-Hand: target(체커보드)이 월드에 고정 → 모든 포즈에서 target의 base 좌표가 동일해야 함

    Returns:
        (평균 편차 mm, 각 포즈별 편차 리스트)
    """
    target_in_base_list = []
    for R_in, t_in, R_tc, t_tc in zip(R_in_list, t_in_list, R_target2cam_list, t_target2cam_list):
        T_in = np.eye(4)
        T_in[:3, :3] = R_in
        T_in[:3, 3] = t_in.flatten()

        T_tc = np.eye(4)
        T_tc[:3, :3] = R_tc
        T_tc[:3, 3] = t_tc.flatten()

        if mode == "eye_to_hand":
            # T_result = T_cam2base
            # T_in = T_base2gripper (역행렬 입력)
            # 체커보드는 gripper에 고정 → T_target_in_gripper = T_in @ T_cam2base @ T_target2cam
            T_target_in_ref = T_in @ T_result @ T_tc
        else:  # eye_in_hand
            # T_result = T_cam2gripper
            # T_in = T_gripper2base
            # 체커보드는 월드에 고정 → T_target_in_base = T_gripper2base @ T_cam2gripper @ T_target2cam
            T_target_in_ref = T_in @ T_result @ T_tc

        target_in_base_list.append(T_target_in_ref[:3, 3])

    target_array = np.array(target_in_base_list)
    mean_pos = target_array.mean(axis=0)
    deviations = np.linalg.norm(target_array - mean_pos, axis=1)
    return float(deviations.mean()), deviations.tolist()


def save_calibration_result(T: np.ndarray, path: str, mode: Optional[str] = None):
    """calibration 결과를 JSON 파일로 저장"""
    result = {
        "transformation_matrix": T.tolist(),
        "tcp_format": homogeneous_to_tcp(T),
    }
    if mode:
        result["mode"] = mode
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Calibration 결과 저장: {path}")


def load_calibration_result(path: str) -> Optional[np.ndarray]:
    """저장된 calibration 결과 로드"""
    try:
        with open(path) as f:
            result = json.load(f)
        return np.array(result["transformation_matrix"])
    except Exception as e:
        logger.error(f"Calibration 결과 로드 실패: {e}")
        return None
