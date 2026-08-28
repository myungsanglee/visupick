"""
CAD 정합/매칭 알고리즘 (순수 모듈 — Qt 무의존)
===============================================
cad_matching_tab.py 안에 UI 와 동거하던 모듈 레벨 알고리즘 22개를 분리했다
(opening_analysis.py 와 같은 원칙: 알고리즘은 앱 없이 단독 실행/튜닝 가능해야 한다).

구성 (자세한 수식/원리는 docs/cad_matching.md):
  - CAD 로드/전처리: load_cad_model, cull_model_to_visible, preprocess_pcd
  - 전역 정합: global_register(FPFH+RANSAC), global_register_fgr(FGR)
  - 정밀 정합: refine_icp
  - 멀티 인스턴스: cad_match_multi_instance(정합→제거→반복),
    cad_match_per_cluster(DBSCAN 클러스터별), remove_inliers_from_scene
  - PPF (opencv-contrib): train_ppf_detector, ppf_match_per_cluster,
    ppf_match_whole_scene(전체장면 voting — DBSCAN 불필요, 권장)
  - 장면 전처리: remove_table_plane, cluster_scene_dbscan, crop_pointcloud_by_2d_roi
  - 자세 → 로봇: object_pose_to_tcp, tool_frame_from_normal,
    suggest_rotation_from_normal, _aligned_tool_frame

UI 쪽(cad_matching_tab.py)은 위젯 값을 읽어 이 함수들을 호출하는 래퍼만 갖는다.
"""

import copy
import time
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

from calibration import tcp_to_homogeneous, homogeneous_to_tcp

# PPF 모듈은 opencv-contrib-python에만 있음. 없으면 호출 측에서 PPF 옵션 자동 비활성.
HAS_PPF = hasattr(cv2, "ppf_match_3d")

logger = logging.getLogger(__name__)


# ============================================================
# 매칭 알고리즘 (Open3D 기반)
# ============================================================


def cull_model_to_visible(
    pcd: o3d.geometry.PointCloud,
    view_axis: str = "+Z",
) -> o3d.geometry.PointCloud:
    """
    주어진 시야 방향에서 보이는 표면만 추출 (Open3D hidden point removal).

    카메라가 객체 좌표계의 view_axis 방향(+Z 등)에서 객체를 본다고 가정.
    예: view_axis="+Z"면 카메라가 +Z 위에 있고 -Z 방향을 봄.

    이게 없으면 model 8000점 중 보이지 않는 바닥/옆면 점들이 매칭을 망친다.
    윗면만 보이는 (위에서 아래로 촬영하는) 빈 픽킹 환경에서 거의 필수.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    center = (bbox_min + bbox_max) / 2.0

    axis_map = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    direction = axis_map.get(view_axis, np.array([0.0, 0.0, 1.0]))

    # 카메라를 객체 중심에서 direction 방향으로 충분히 멀리
    camera_pos = (center + direction * diag * 5.0).tolist()
    radius = diag * 100.0

    try:
        _, pt_map = pcd.hidden_point_removal(camera_pos, radius)
        if len(pt_map) < 100:
            logger.warning(f"hidden_point_removal 후 점 부족: {len(pt_map)} (cull 무시하고 원본 사용)")
            return pcd
        result = pcd.select_by_index(pt_map)

        # 법선을 카메라 방향(=+axis 외향)으로 일관되게 정렬.
        # 이게 없으면 estimate_normals 결과가 +/- 무작위라 FPFH descriptor가 망가져서
        # RANSAC fitness=0이 나옴.
        result_diag = float(np.linalg.norm(np.asarray(result.get_axis_aligned_bounding_box().get_extent())))
        result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.5, result_diag * 0.05), max_nn=30))
        result.orient_normals_towards_camera_location(np.array(camera_pos))
        return result
    except Exception as e:
        logger.warning(f"hidden_point_removal 실패: {e}")
        return pcd


def load_cad_model(path: str, n_sample_points: int = 8000) -> Optional[Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]]:
    """
    CAD 파일을 Open3D로 로드.

    STL/OBJ는 mesh로 읽고 표면 균등 샘플링으로 포인트 클라우드 생성.
    PLY는 이미 포인트 클라우드일 수 있음.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix in (".stl", ".obj"):
            mesh = o3d.io.read_triangle_mesh(path)
            if not mesh.has_triangles():
                logger.error(f"메시에 삼각형이 없음: {path}")
                return None
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_poisson_disk(n_sample_points)
            return pcd, mesh
        elif suffix == ".ply":
            # PLY는 mesh 또는 pcd일 수 있음
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh.has_triangles():
                mesh.compute_vertex_normals()
                pcd = mesh.sample_points_poisson_disk(n_sample_points)
                return pcd, mesh
            else:
                pcd = o3d.io.read_point_cloud(path)
                return pcd, None
        else:
            logger.error(f"지원하지 않는 포맷: {suffix}")
            return None
    except Exception as e:
        logger.error(f"CAD 로드 실패: {e}")
        return None


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    다운샘플 + 법선 + FPFH descriptor.

    voxel_down_sample은 입력 pcd의 normal을 평균내서 보존함.
    호출 전에 normal이 카메라 방향으로 정렬되어 있어야 FPFH가 제대로 작동.
    이미 normal이 있으면 재추정하지 않는다 (정렬 결과 보존).
    """
    down = pcd.voxel_down_sample(voxel_size)
    if not down.has_normals():
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return down, fpfh


def global_register_fgr(
    model_down: o3d.geometry.PointCloud,
    scene_down: o3d.geometry.PointCloud,
    model_fpfh: o3d.pipelines.registration.Feature,
    scene_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Fast Global Registration (Zhou et al. 2016).

    RANSAC 대비 장점:
      - 결정적 (같은 입력 → 같은 결과). 재현성 ↑
      - 보통 5~10배 빠름
      - Truncated least squares 손실로 노이즈 robust

    한계: FPFH-based라 cull/visible 가정의 본질적 한계는 RANSAC와 동일.
          무작위 자세에는 PPF가 여전히 더 견고.
    """
    # FGR이 RANSAC보다 더 빡빡한 tuple constraint를 강제하므로 파라미터를 매우 관대하게 설정.
    # 그래도 안 되면 RANSAC을 권장 (FGR은 알고리즘 특성상 일부 시나리오에서 본질적으로 약함).
    distance_threshold = voxel_size * 2.0
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            decrease_mu=True,
            iteration_number=128,
            maximum_tuple_count=5000,  # 더 많은 tuple 후보
            tuple_scale=0.85,  # 더 관대한 tuple test (기본 0.95)
        ),
    )


def global_register(
    model_down: o3d.geometry.PointCloud,
    scene_down: o3d.geometry.PointCloud,
    model_fpfh: o3d.pipelines.registration.Feature,
    scene_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    RANSAC 기반 글로벌 매칭 → 초기 자세

    Checker 3개로 잘못된 매칭 강하게 차단:
      - EdgeLength: 모델 내부 에지 비율 보존
      - Distance: max_correspondence_distance 안에 들어와야 함
      - Normal: 법선 방향이 비슷해야 함 (윗면-바닥 같은 잘못된 페어 차단)
    """
    distance_threshold = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(30.0)),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.9999),
    )


def refine_icp(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Two-pass Point-to-plane ICP로 정밀화.

    Open3D 표준 튜토리얼은 voxel * 0.4를 쓰지만, RANSAC 초기 자세가 1~3mm 어긋나
    있는 게 일반적이라 그 값으론 inlier가 거의 안 잡혀 fitness가 낮게 나옴.
    1차는 관대(voxel * 1.5)하게 잡아서 자세를 끌어당기고,
    2차에서 살짝 더 좁힌 거리(voxel * 1.0)로 정밀화한다.
    fitness 평가도 2차 거리 기준이므로 일관됨.
    """
    if not scene_pcd.has_normals():
        scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))

    # 1차: 관대한 거리로 자세 끌어당기기
    coarse = o3d.pipelines.registration.registration_icp(
        model_pcd,
        scene_pcd,
        voxel_size * 1.5,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    # 2차: 좀 더 좁힌 거리로 fitness 평가 (너무 좁히면 옛 문제 재발)
    fine = o3d.pipelines.registration.registration_icp(
        model_pcd,
        scene_pcd,
        voxel_size * 1.0,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return fine


def remove_inliers_from_scene(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    T: np.ndarray,
    radius: float,
) -> o3d.geometry.PointCloud:
    """
    매칭된 영역의 scene 점들을 제거 (다음 인스턴스 검색을 위해).

    이전 구현은 변환된 model 점마다 Python 루프로 radius 검색 + set 누적이라
    느렸다. Open3D `compute_point_cloud_distance`(내부 C++ KDTree)로 scene 각
    점→model 최근접 거리를 한 번에 구해 radius 이내를 제거 — 동작 동일, 훨씬 빠름.
    """
    scene_pts = np.asarray(scene_pcd.points)
    if len(scene_pts) == 0:
        return scene_pcd

    # model을 scene 좌표계로 (deepcopy 대신 점만 변환한 가벼운 사본)
    model_pts = np.asarray(model_pcd.points)
    model_t = model_pts @ T[:3, :3].T + T[:3, 3]
    model_in_scene = o3d.geometry.PointCloud()
    model_in_scene.points = o3d.utility.Vector3dVector(model_t)

    dists = np.asarray(scene_pcd.compute_point_cloud_distance(model_in_scene))
    keep_idx = np.where(dists > radius)[0]
    if len(keep_idx) == len(scene_pts):
        return scene_pcd
    return scene_pcd.select_by_index(keep_idx.tolist())


def cad_match_multi_instance(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_instances: int = 5,
    fitness_threshold: float = 0.2,
    ransac_attempts: int = 5,
    use_fgr: bool = False,
    progress_cb=None,
) -> Tuple[List[Dict], List[str]]:
    """
    scene 안에서 model에 해당하는 객체를 여러 개 찾음.

    재현성과 안정성을 위해:
      - 글로벌 시드 고정 (매번 같은 결과)
      - 각 인스턴스마다 RANSAC을 ransac_attempts번 시도해 best fitness 채택
        (RANSAC randomness + 윗면 대칭 모호성 보정)

    Returns:
        (instances, debug_log)
    """
    # 결정적 결과를 위해 Open3D 글로벌 RNG 시드 고정
    try:
        o3d.utility.random.seed(42)
    except AttributeError:
        pass  # 구 버전 호환

    instances: List[Dict] = []
    debug_log: List[str] = []

    # 모델은 한 번만 다운샘플 + FPFH 계산
    model_down, model_fpfh = preprocess_pcd(model_pcd, voxel_size)
    if not model_pcd.has_normals():
        model_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    debug_log.append(f"Model: 원본 {len(model_pcd.points)}점 → 다운샘플 {len(model_down.points)}점 (voxel={voxel_size}mm)")

    remaining = scene_pcd
    debug_log.append(f"Scene 시작: {len(remaining.points)}점")
    if use_fgr:
        debug_log.append("글로벌 매칭: FGR (결정적, 1회 시도)")
    else:
        debug_log.append(f"글로벌 매칭: RANSAC ({ransac_attempts}회 시도, best 채택, seed 고정)")

    for i in range(max_instances):
        if progress_cb:
            progress_cb(i, max_instances, f"매칭 {i + 1}/{max_instances} 시도 중...")

        if len(remaining.points) < len(model_down.points) // 2:
            debug_log.append(f"[시도 {i + 1}] 남은 scene 점 부족 ({len(remaining.points)}) → 종료")
            break

        scene_down, scene_fpfh = preprocess_pcd(remaining, voxel_size)
        if len(scene_down.points) < 30:
            debug_log.append(f"[시도 {i + 1}] 다운샘플 후 점 부족 ({len(scene_down.points)}) → 종료")
            break

        # 1) 글로벌 매칭
        if use_fgr:
            best_ransac = global_register_fgr(model_down, scene_down, model_fpfh, scene_fpfh, voxel_size)
            debug_log.append(f"[시도 {i + 1}] scene_down={len(scene_down.points)}점, FGR fitness={best_ransac.fitness:.3f}")
        else:
            ransac_results = []
            for k in range(ransac_attempts):
                r = global_register(model_down, scene_down, model_fpfh, scene_fpfh, voxel_size)
                ransac_results.append(r)
            best_ransac = max(ransac_results, key=lambda r: r.fitness)
            fit_strs = ", ".join(f"{r.fitness:.3f}" for r in ransac_results)
            debug_log.append(f"[시도 {i + 1}] scene_down={len(scene_down.points)}점, RANSAC[{fit_strs}] → best={best_ransac.fitness:.3f}")

        if best_ransac.fitness == 0:
            debug_log.append(f"  → RANSAC 실패 (descriptor 매칭 안됨), 종료")
            break

        # 2) ICP 정밀화 (best RANSAC 결과를 초기값으로)
        result_icp = refine_icp(model_pcd, remaining, best_ransac.transformation, voxel_size)
        fitness = float(result_icp.fitness)
        rmse = float(result_icp.inlier_rmse)
        debug_log.append(f"  ICP: fitness={fitness:.3f}, RMSE={rmse:.3f}mm")

        if fitness < fitness_threshold:
            debug_log.append(f"  → fitness {fitness:.3f} < 임계값 {fitness_threshold}, 종료")
            break

        instances.append(
            {
                "transformation": np.asarray(result_icp.transformation).copy(),
                "fitness": fitness,
                "rmse": rmse,
            }
        )
        debug_log.append(f"  ✓ 인스턴스 #{len(instances)} 채택")

        # 3) 매칭된 점 제거
        remaining = remove_inliers_from_scene(remaining, model_pcd, result_icp.transformation, voxel_size * 2.0)

    return instances, debug_log


def remove_table_plane(
    scene_pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 5.0,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[o3d.geometry.PointCloud, Optional[Tuple[float, float, float, float]], int]:
    """
    Scene에서 가장 큰 평면(작업대)을 RANSAC으로 검출하고 그 점들을 제거.

    이게 없으면 작업대 표면 점들이 한 거대 클러스터를 만들어서 그 안에 객체가 묻힘.

    Args:
        distance_threshold: 평면에 속한다고 인정할 최대 거리(mm). 작업대 평탄도와 노이즈에 따라 2~10mm.

    Returns:
        (평면 제외한 PointCloud, 평면 방정식 (a,b,c,d), 제거된 점 수)
        평면이 없거나 점 부족 시 원본 그대로 + None + 0 반환.
    """
    if len(scene_pcd.points) < ransac_n + 10:
        return scene_pcd, None, 0
    try:
        plane_model, inliers = scene_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
    except Exception as e:
        logger.warning(f"평면 검출 실패: {e}")
        return scene_pcd, None, 0

    n_inliers = len(inliers)
    if n_inliers == 0:
        return scene_pcd, None, 0

    outlier = scene_pcd.select_by_index(inliers, invert=True)
    return outlier, tuple(float(x) for x in plane_model), n_inliers


def cluster_scene_dbscan(
    scene_pcd: o3d.geometry.PointCloud,
    eps: float = 15.0,
    min_points: int = 50,
) -> List[o3d.geometry.PointCloud]:
    """
    DBSCAN으로 scene을 객체 단위 클러스터로 분리.

    Args:
        eps: 같은 클러스터로 묶일 점 사이 최대 거리 (mm).
             객체 크기/노이즈에 따라 5~30mm 권장.
        min_points: 클러스터 최소 점 수. 노이즈 클러스터 차단용.

    Returns:
        각 클러스터를 별도 PointCloud로 담은 리스트 (점 수 내림차순).
        노이즈 점(label=-1)은 무시.
    """
    if len(scene_pcd.points) < min_points:
        return []

    labels = np.array(scene_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    n_clusters = int(labels.max()) + 1 if labels.size > 0 else 0
    if n_clusters <= 0:
        return []

    sized: List[Tuple[int, o3d.geometry.PointCloud]] = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) >= min_points:
            sized.append((len(idx), scene_pcd.select_by_index(idx.tolist())))

    sized.sort(key=lambda x: -x[0])  # 큰 클러스터부터
    return [c[1] for c in sized]


def cad_match_per_cluster(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    eps: float = 15.0,
    min_points: int = 50,
    fitness_threshold: float = 0.2,
    ransac_attempts: int = 5,
    use_fgr: bool = False,
    progress_cb=None,
) -> Tuple[List[Dict], List[str]]:
    """
    Scene을 DBSCAN으로 클러스터 분리한 뒤, 각 클러스터에 대해 한 번씩 매칭.

    멀티 인스턴스 RANSAC(반복 매칭)과 비교한 장점:
      - 객체 경계가 사전에 분리되어 RANSAC이 두 객체 사이를 매칭하는 일이 없음
      - "매칭된 점 제거" 단계 불필요 (각 클러스터가 독립)
      - 멀티 인스턴스가 자연스럽게 처리 (각 클러스터 = 한 후보)

    한계:
      - 맞닿거나 겹친 객체는 한 클러스터로 묶임 → 객체 분리 실패
      - 평평한 작업대 점들이 한 거대 클러스터를 만들면 그 안의 객체와 섞임
    """
    try:
        o3d.utility.random.seed(42)
    except AttributeError:
        pass

    instances: List[Dict] = []
    debug_log: List[str] = []

    clusters = cluster_scene_dbscan(scene_pcd, eps=eps, min_points=min_points)
    debug_log.append(f"DBSCAN: scene {len(scene_pcd.points)}점 → 클러스터 {len(clusters)}개 (eps={eps}mm, min_pts={min_points})")
    if not clusters:
        debug_log.append("클러스터 없음 → 종료. eps를 늘리거나 min_pts를 줄여보세요.")
        return instances, debug_log

    # 모델 한 번만 전처리
    model_down, model_fpfh = preprocess_pcd(model_pcd, voxel_size)
    if not model_pcd.has_normals():
        model_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    debug_log.append(f"Model: 원본 {len(model_pcd.points)}점 → 다운샘플 {len(model_down.points)}점 (voxel={voxel_size}mm)")
    if use_fgr:
        debug_log.append("글로벌 매칭: FGR (결정적, 1회 시도)")
    else:
        debug_log.append(f"글로벌 매칭: RANSAC ({ransac_attempts}회 시도, best 채택, seed 고정)")

    for ci, cluster in enumerate(clusters):
        if progress_cb:
            progress_cb(ci, len(clusters), f"클러스터 {ci + 1}/{len(clusters)} 매칭 중...")

        if len(cluster.points) < len(model_down.points) // 2:
            debug_log.append(f"[클러스터 {ci + 1}] {len(cluster.points)}점, 모델 절반 미만 → 스킵")
            continue

        scene_down, scene_fpfh = preprocess_pcd(cluster, voxel_size)
        if len(scene_down.points) < 30:
            debug_log.append(f"[클러스터 {ci + 1}] 다운샘플 후 점 부족 ({len(scene_down.points)}) → 스킵")
            continue

        # 글로벌 매칭
        if use_fgr:
            best_ransac = global_register_fgr(model_down, scene_down, model_fpfh, scene_fpfh, voxel_size)
            debug_log.append(
                f"[클러스터 {ci + 1}] {len(cluster.points)}점, scene_down={len(scene_down.points)}점, " f"FGR fitness={best_ransac.fitness:.3f}"
            )
        else:
            ransac_results = []
            for k in range(ransac_attempts):
                r = global_register(model_down, scene_down, model_fpfh, scene_fpfh, voxel_size)
                ransac_results.append(r)
            best_ransac = max(ransac_results, key=lambda r: r.fitness)
            fit_strs = ", ".join(f"{r.fitness:.3f}" for r in ransac_results)
            debug_log.append(
                f"[클러스터 {ci + 1}] {len(cluster.points)}점, scene_down={len(scene_down.points)}점, "
                f"RANSAC[{fit_strs}] → best={best_ransac.fitness:.3f}"
            )

        if best_ransac.fitness == 0:
            debug_log.append(f"  → RANSAC 실패, 스킵")
            continue

        # ICP 정밀화
        result_icp = refine_icp(model_pcd, cluster, best_ransac.transformation, voxel_size)
        fitness = float(result_icp.fitness)
        rmse = float(result_icp.inlier_rmse)
        debug_log.append(f"  ICP: fitness={fitness:.3f}, RMSE={rmse:.3f}mm")

        if fitness < fitness_threshold:
            debug_log.append(f"  → fitness {fitness:.3f} < 임계값 {fitness_threshold}, 스킵")
            continue

        instances.append(
            {
                "transformation": np.asarray(result_icp.transformation).copy(),
                "fitness": fitness,
                "rmse": rmse,
                "cluster_id": ci,
                "cluster_size": len(cluster.points),
            }
        )
        debug_log.append(f"  ✓ 인스턴스 #{len(instances)} 채택 (cluster {ci + 1})")

    return instances, debug_log


def pcd_to_ppf_format(pcd: o3d.geometry.PointCloud, normal_radius: Optional[float] = None) -> np.ndarray:
    """
    Open3D PointCloud → OpenCV PPF 입력 형식 (Nx6 float32: [x,y,z,nx,ny,nz]).

    PPF는 normal 필수. 없으면 자동 추정. 일관성 정렬은 호출자 책임 (학습 시 한 번,
    scene 측은 카메라 방향으로 정렬되어 있으면 좋음).
    """
    pts = np.asarray(pcd.points, dtype=np.float32)
    if not pcd.has_normals():
        if normal_radius is None:
            bbox = pcd.get_axis_aligned_bounding_box()
            diag = float(np.linalg.norm(bbox.get_extent()))
            normal_radius = max(0.5, diag * 0.05)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack([pts, nrm]).astype(np.float32)


def _orient_normals_outward(pcd: o3d.geometry.PointCloud) -> None:
    """
    객체 중심 기준으로 normal을 외향으로 정렬 (in-place).

    `orient_normals_consistent_tangent_plane`은 KNN graph + MST 기반이라
    8000점에서 분 단위가 걸림. 거의 凸한 객체(차단기 등)에는 단순 중심 기반
    외향 정렬로 충분하고 매우 빠름.
    """
    pts = np.asarray(pcd.points)
    if not pcd.has_normals() or len(pts) == 0:
        return
    nrm = np.asarray(pcd.normals)
    center = pts.mean(axis=0)
    to_pt = pts - center
    dots = np.einsum("ij,ij->i", nrm, to_pt)
    flip_mask = dots < 0
    if flip_mask.any():
        nrm = nrm.copy()
        nrm[flip_mask] = -nrm[flip_mask]
        pcd.normals = o3d.utility.Vector3dVector(nrm)


def train_ppf_detector(
    model_pcd: o3d.geometry.PointCloud,
    relative_sampling_step: float = 0.04,
    relative_distance_step: float = 0.05,
    num_angles: int = 30,
    max_points_for_train: int = 2500,
):
    """
    OpenCV PPF detector 학습. 같은 CAD 모델 매칭 여러 번 시 한 번만 호출.

    학습 시간 폭증을 막기 위한 두 가지 안전장치:
      - max_points_for_train 초과 시 자동 다운샘플 (학습은 점쌍 N² 비례)
      - normal 정렬은 객체 중심 기반 외향 정렬 (consistent_tangent_plane 대비 ~수십배 빠름)

    relative_sampling_step: 모델 직경의 비율로 PPF 내부 다운샘플 강도.
        0.04 (4%)는 정확도/속도 균형. 더 빠르게: 0.06. 더 정밀하게: 0.025.
    relative_distance_step: 해시 거리 양자화 (기본 5%).
    """
    if not HAS_PPF:
        raise RuntimeError("PPF 모듈 미사용. opencv-contrib-python 설치 필요.")

    model_copy = copy.deepcopy(model_pcd)

    # 큰 모델은 학습 시간 폭증 → 자동 다운샘플
    n_orig = len(model_copy.points)
    if n_orig > max_points_for_train:
        bbox = model_copy.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(bbox.get_extent()))
        # 표면적이 알려지지 않았으니 N^(1/2) 비례로 voxel 추정 (대충)
        ratio = (n_orig / max_points_for_train) ** 0.5
        target_voxel = max(0.5, diag * 0.025 * ratio)
        model_copy = model_copy.voxel_down_sample(target_voxel)
        logger.info(f"PPF 학습용 다운샘플: {n_orig}→{len(model_copy.points)}점 (voxel={target_voxel:.2f}mm)")

    # Normal 추정 (없으면)
    if not model_copy.has_normals():
        bbox = model_copy.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(bbox.get_extent()))
        model_copy.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.5, diag * 0.05), max_nn=30))

    # 외향 정렬 (객체 중심 기반, 매우 빠름)
    _orient_normals_outward(model_copy)

    model_data = pcd_to_ppf_format(model_copy)
    detector = cv2.ppf_match_3d.PPF3DDetector(
        relative_sampling_step,
        relative_distance_step,
        num_angles,
    )
    detector.trainModel(model_data)
    # model_copy도 함께 반환 (Open3D ICP 정밀화에 사용)
    return detector, model_data, model_copy


def ppf_match_per_cluster(
    scene_pcd: o3d.geometry.PointCloud,
    detector,
    model_data: np.ndarray,
    model_o3d: o3d.geometry.PointCloud,
    eps: float = 15.0,
    min_points: int = 100,
    min_votes: int = 100,
    relative_scene_sample_step: float = 1.0 / 40.0,
    relative_scene_distance: float = 0.03,
    n_top_candidates: int = 5,
    n_show_per_cluster: int = 1,
    progress_cb=None,
) -> Tuple[List[Dict], List[str]]:
    """
    Scene을 DBSCAN으로 클러스터 분리한 뒤, 각 클러스터마다 PPF voting + Open3D ICP.

    PPF는 무작위 자세 voting에 강건하지만, OpenCV의 자체 ICP는 평면 객체에서
    슬라이드 모호성이 약함. 그래서 PPF voting으로 초기 자세를 얻은 뒤,
    검증된 두 단계 point-to-plane ICP(`refine_icp`)로 정밀화.

    각 클러스터:
      1. PPF voting → 자세 후보 N개
      2. 각 후보를 Open3D ICP로 정밀화 → fitness/RMSE 산출
      3. 상위 K개를 모두 인스턴스로 출력 (사용자가 시각적 선택)
    """
    if not HAS_PPF:
        return [], ["PPF 모듈 미사용 (opencv-contrib-python 설치 필요)"]

    instances: List[Dict] = []
    debug_log: List[str] = []

    clusters = cluster_scene_dbscan(scene_pcd, eps=eps, min_points=min_points)
    debug_log.append(f"DBSCAN: scene {len(scene_pcd.points)}점 → 클러스터 {len(clusters)}개")
    if not clusters:
        debug_log.append("클러스터 없음 → 종료")
        return instances, debug_log

    # Model 크기 기반 결정값들
    model_pts = model_data[:, :3]
    model_diag = float(np.linalg.norm(model_pts.max(axis=0) - model_pts.min(axis=0)))
    scene_normal_radius = max(0.5, model_diag * 0.05)
    # Open3D ICP의 거리 임계값 베이스. relative_sampling_step과 유사한 스케일.
    voxel_for_icp = max(0.5, model_diag * 0.025)
    debug_log.append(
        f"PPF model: {len(model_data)}점 (학습 완료, model diag={model_diag:.1f}mm, "
        f"scene normal r={scene_normal_radius:.2f}mm, ICP voxel={voxel_for_icp:.2f}mm)"
    )

    for ci, cluster in enumerate(clusters):
        if progress_cb:
            progress_cb(ci, len(clusters), f"PPF 매칭 클러스터 {ci + 1}/{len(clusters)}")

        # PPF는 model/scene normal이 같은 convention이어야 descriptor 일관됨.
        # 빈 픽킹은 모든 보이는 표면이 카메라 향함 = 객체 외향 → 카메라 원점 정렬.
        # normal radius는 cluster 크기 대신 model 크기 기준 (위에서 미리 계산)
        # → cluster에 잡음 섞여도 객체 디테일 스케일 유지.
        cluster = copy.deepcopy(cluster)
        cluster.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scene_normal_radius, max_nn=30))
        cluster.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

        scene_data = pcd_to_ppf_format(cluster)
        if len(scene_data) < 100:
            debug_log.append(f"[클러스터 {ci + 1}] {len(scene_data)}점, 너무 적음 → 스킵")
            continue

        try:
            results = detector.match(
                scene_data,
                relative_scene_sample_step,
                relative_scene_distance,
            )
        except Exception as e:
            debug_log.append(f"[클러스터 {ci + 1}] PPF 매칭 오류: {e}")
            continue

        if not results:
            debug_log.append(f"[클러스터 {ci + 1}] {len(scene_data)}점, PPF voting 결과 없음")
            continue

        n_use = min(n_top_candidates, len(results))
        candidates = list(results[:n_use])
        votes_before = [int(c.numVotes) for c in candidates]
        debug_log.append(f"[클러스터 {ci + 1}] {len(scene_data)}점, PPF 후보 {len(results)}개, " f"상위 {n_use}개 votes={votes_before}")

        # 각 PPF 후보를 Open3D point-to-plane ICP로 정밀화 (평면 슬라이드 약점 보완)
        refined_results = []
        for cand_idx, cand in enumerate(candidates):
            init_T = np.asarray(cand.pose, dtype=np.float64)
            votes = int(getattr(cand, "numVotes", 0))
            try:
                icp_result = refine_icp(model_o3d, cluster, init_T, voxel_for_icp)
                refined_results.append(
                    {
                        "transformation": np.asarray(icp_result.transformation, dtype=np.float64).copy(),
                        "fitness": float(icp_result.fitness),
                        "rmse": float(icp_result.inlier_rmse),
                        "votes": votes,
                    }
                )
            except Exception as e:
                debug_log.append(f"  후보 {cand_idx + 1} ICP 실패 (skip): {e}")

        if not refined_results:
            debug_log.append(f"  → ICP 후 유효 후보 없음")
            continue

        # Open3D ICP fitness 큰 순으로 정렬 (높을수록 더 많은 model 점이 scene에 매칭됨).
        # 평면 슬라이드된 후보는 fitness가 낮게 나와 자동으로 밀려남.
        refined_results.sort(key=lambda r: -r["fitness"])

        n_show = min(n_show_per_cluster, len(refined_results))
        accepted = 0
        for k in range(n_show):
            cand = refined_results[k]
            if cand["votes"] < min_votes:
                debug_log.append(f"  후보 {k + 1}: votes={cand['votes']} < {min_votes}, 스킵")
                continue
            instances.append(
                {
                    "transformation": cand["transformation"],
                    "fitness": cand["fitness"],  # Open3D fitness (0~1, 클수록 좋음)
                    "rmse": cand["rmse"],  # mm
                    "votes": cand["votes"],
                    "cluster_id": ci,
                    "cluster_size": len(cluster.points),
                    "rank_in_cluster": k + 1,
                }
            )
            accepted += 1
            debug_log.append(
                f"  후보 {k + 1}: votes={cand['votes']}, ICP fitness={cand['fitness']:.3f}, "
                f"RMSE={cand['rmse']:.3f}mm → 인스턴스 #{len(instances)} 채택"
            )

        if accepted == 0:
            debug_log.append(f"  → 임계값 통과 후보 없음")

    return instances, debug_log


def ppf_match_whole_scene(
    scene_pcd: o3d.geometry.PointCloud,
    detector,
    model_data: np.ndarray,
    model_o3d: o3d.geometry.PointCloud,
    min_votes: int = 100,
    relative_scene_sample_step: float = 1.0 / 40.0,
    relative_scene_distance: float = 0.03,
    n_candidates: int = 60,
    max_instances: int = 10,
    fitness_threshold: float = 0.15,
    nms_dist_frac: float = 0.5,
    scene_voxel: float = 0.0,
    progress_cb=None,
) -> Tuple[List[Dict], List[str]]:
    """DBSCAN 없이 **전체 장면에 PPF voting 1회** → 다중 인스턴스 6D 자세 (상용 방식).

    per-cluster 와 달리 장면을 미리 나누지 않아, 쌓여 붙은/부분 가림 객체에 강건하다.
    절차:
      1. (선택) 장면 voxel 다운샘플 → normal 추정(카메라 향해 정렬) → Nx6 변환.
      2. detector.match(전체 장면) → voting 후보 다수 (같은 인스턴스에 여러 후보 나옴).
      3. **pre-ICP NMS**: raw voting 자세로 먼저 중복 제거 → 서로 다른 인스턴스 후보만 남김
         (ICP 횟수를 확 줄여 속도↑, 놓침 위험은 낮음).
      4. 남은 후보를 Open3D point-to-plane ICP 로 정밀화 → votes/fitness 임계 통과분을
         fitness 순 post-NMS → 최대 max_instances 개.

    속도 튜닝:
      scene_voxel(>0): 매칭 전 장면을 이 해상도로 다운샘플 (voting 비용의 최대 지렛대).
      relative_scene_sample_step: voting 기준점 샘플링 (키우면 빠름).
    각 단계 소요 시간(ms)을 debug_log 에 남겨 병목(voting vs ICP)을 확인한다.
    """
    if not HAS_PPF:
        return [], ["PPF 모듈 미사용 (opencv-contrib-python 설치 필요)"]

    instances: List[Dict] = []
    debug_log: List[str] = []

    model_pts = model_data[:, :3]
    model_diag = float(np.linalg.norm(model_pts.max(axis=0) - model_pts.min(axis=0)))
    model_centroid = model_pts.mean(axis=0)
    scene_normal_radius = max(0.5, model_diag * 0.05)
    voxel_for_icp = max(0.5, model_diag * 0.025)
    nms_dist = model_diag * nms_dist_frac

    # --- 준비: (선택)다운샘플 + normal + 변환 ---
    t0 = time.perf_counter()
    scene = copy.deepcopy(scene_pcd)
    n_before = len(scene.points)
    if scene_voxel and scene_voxel > 0:
        scene = scene.voxel_down_sample(scene_voxel)
    scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scene_normal_radius, max_nn=30))
    scene.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    scene_data = pcd_to_ppf_format(scene)
    prep_ms = (time.perf_counter() - t0) * 1000.0
    ds_msg = f"{n_before}→{len(scene.points)}점(voxel={scene_voxel:.2f}mm)" if scene_voxel > 0 else f"{len(scene.points)}점(다운샘플 없음)"
    debug_log.append(f"전체장면 PPF: scene {ds_msg}, model diag={model_diag:.1f}mm, " f"ICP voxel={voxel_for_icp:.2f}mm, NMS 거리={nms_dist:.1f}mm")
    if len(scene_data) < 100:
        debug_log.append("장면 점이 너무 적음 → 종료")
        return instances, debug_log

    # --- voting ---
    if progress_cb:
        progress_cb(0, 3, "전체장면 PPF voting...")
    t0 = time.perf_counter()
    try:
        results = detector.match(scene_data, relative_scene_sample_step, relative_scene_distance)
    except Exception as e:
        debug_log.append(f"PPF 매칭 오류: {e}")
        return instances, debug_log
    vote_ms = (time.perf_counter() - t0) * 1000.0
    if not results:
        debug_log.append(f"PPF voting 결과 없음 (voting {vote_ms:.0f}ms)")
        return instances, debug_log

    # --- pre-ICP NMS: raw voting 자세로 중복 제거 (ICP 횟수 절감) ---
    n_use = min(n_candidates, len(results))
    raw = []
    for cand in results[:n_use]:
        votes = int(getattr(cand, "numVotes", 0))
        if votes < min_votes:
            continue
        T0 = np.asarray(cand.pose, dtype=np.float64)
        raw.append({"T0": T0, "votes": votes, "center0": (T0 @ np.append(model_centroid, 1.0))[:3]})
    raw.sort(key=lambda r: -r["votes"])
    kept_raw = []
    for r in raw:
        if any(np.linalg.norm(r["center0"] - k["center0"]) < nms_dist for k in kept_raw):
            continue
        kept_raw.append(r)
        if len(kept_raw) >= max_instances * 2:  # ICP 검증에서 일부 탈락 대비 여유
            break
    debug_log.append(f"PPF 후보 {len(results)}개 → votes≥{min_votes} {len(raw)}개 → pre-ICP NMS 후 {len(kept_raw)}개 ICP")

    # --- ICP 정밀화 (pre-NMS로 걸러진 후보만) ---
    t0 = time.perf_counter()
    refined: List[Dict] = []
    for idx, r in enumerate(kept_raw):
        if progress_cb:
            progress_cb(1, 3, f"ICP 정밀화 {idx + 1}/{len(kept_raw)}")
        try:
            icp = refine_icp(model_o3d, scene, r["T0"], voxel_for_icp)
        except Exception:
            continue
        T = np.asarray(icp.transformation, dtype=np.float64)
        if float(icp.fitness) < fitness_threshold:
            continue
        refined.append(
            {
                "transformation": T.copy(),
                "fitness": float(icp.fitness),
                "rmse": float(icp.inlier_rmse),
                "votes": r["votes"],
                "center": (T @ np.append(model_centroid, 1.0))[:3],
            }
        )
    icp_ms = (time.perf_counter() - t0) * 1000.0

    if not refined:
        debug_log.append(f"fitness≥{fitness_threshold} 통과 후보 없음 | ⏱ 준비 {prep_ms:.0f} / voting {vote_ms:.0f} / ICP {icp_ms:.0f}ms")
        return instances, debug_log

    # --- post-ICP NMS: ICP 로 중심이 이동했을 수 있어 다시 한 번 (fitness 우선) ---
    t0 = time.perf_counter()
    refined.sort(key=lambda r: -r["fitness"])
    if progress_cb:
        progress_cb(2, 3, "중복 제거(NMS)...")
    for cand in refined:
        if len(instances) >= max_instances:
            break
        if any(np.linalg.norm(cand["center"] - acc["center"]) < nms_dist for acc in instances):
            continue
        instances.append(cand)
        debug_log.append(
            f"인스턴스 #{len(instances)}: votes={cand['votes']}, fitness={cand['fitness']:.3f}, "
            f"RMSE={cand['rmse']:.3f}mm, 중심=({cand['center'][0]:.0f},{cand['center'][1]:.0f},{cand['center'][2]:.0f})"
        )
    nms_ms = (time.perf_counter() - t0) * 1000.0

    debug_log.append(
        f"→ 최종 {len(instances)}개 인스턴스 | ⏱ 준비 {prep_ms:.0f} / voting {vote_ms:.0f} / "
        f"ICP {icp_ms:.0f}({len(kept_raw)}회) / NMS {nms_ms:.0f}ms"
    )
    return instances, debug_log


def crop_pointcloud_by_2d_roi(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    roi_2d: Tuple[int, int, int, int],
) -> Optional[o3d.geometry.PointCloud]:
    """
    (H, W, 3) 카메라 포인트 클라우드에서 2D 픽셀 ROI 영역만 잘라
    Open3D PointCloud로 반환. NaN은 제거.
    """
    h, w = xyz.shape[:2]
    x1, y1, x2, y2 = roi_2d
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    pts = xyz[y1:y2, x1:x2].reshape(-1, 3)
    valid = ~np.any(np.isnan(pts), axis=1)
    pts = pts[valid]
    if len(pts) < 50:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    if rgb is not None:
        colors = rgb[y1:y2, x1:x2].reshape(-1, 3)[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 법선을 Zivid 카메라 원점 방향으로 일관되게 정렬 (FPFH descriptor가 정상 작동하려면 필수)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=20.0, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    return pcd


# ============================================================
# 6D 자세 → 로봇 TCP 자세 변환
# ============================================================


def object_pose_to_tcp(
    T_object_cam: np.ndarray,
    T_calib: np.ndarray,
    calib_mode: str,
    current_tcp: Optional[Dict[str, float]],
    grasp_axis: str = "Z",
    grasp_flip: bool = True,
    grasp_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    grasp_rotation_abc_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Dict[str, float]]:
    """
    6D 객체 자세 → KUKA TCP 자세

    grasp_axis: 객체 좌표계의 어느 축이 "위쪽"(잡는 방향과 평행)인지 지정
    grasp_flip: True면 Tool +Z가 객체 -axis 방향(위에서 아래로 잡음).
                False면 Tool +Z가 객체 +axis 방향.
    grasp_offset_xyz: 객체 좌표계 기준 (dx, dy, dz)mm — 잡는 점이 원점이 아닐 때 보정
    grasp_rotation_abc_deg: 잡기 축 정렬 후 Tool 좌표계 기준 추가 회전 (ZYX intrinsic, KUKA 방식)
                            A=yaw(Tool +Z), B=pitch(Tool +Y), C=roll(Tool +X). (0,0,0)이면 보정 없음.
    """
    # 1. 카메라 좌표계 → 베이스 좌표계
    if calib_mode == "eye_to_hand":
        T_object_base = T_calib @ T_object_cam
    elif calib_mode == "eye_in_hand":
        if current_tcp is None:
            logger.error("Eye-in-Hand인데 현재 TCP가 없음")
            return None
        T_g2b = tcp_to_homogeneous(current_tcp)
        T_object_base = T_g2b @ T_calib @ T_object_cam
    else:
        return None

    # 2. 잡는 점 오프셋 적용 (객체 좌표계 기준)
    T_offset = np.eye(4)
    T_offset[:3, 3] = np.array(grasp_offset_xyz, dtype=float)
    T_grasp_base = T_object_base @ T_offset

    def _apply_tool_rotation(T_tcp_base: np.ndarray) -> Dict[str, float]:
        """Tool 좌표계 기준 ABC 회전 보정 적용 후 KUKA TCP dict 반환."""
        if any(abs(a) > 1e-6 for a in grasp_rotation_abc_deg):
            a_rad, b_rad, c_rad = np.radians(grasp_rotation_abc_deg)
            R_correction = Rotation.from_euler("ZYX", [a_rad, b_rad, c_rad]).as_matrix()
            T_correction = np.eye(4)
            T_correction[:3, :3] = R_correction
            return homogeneous_to_tcp(T_tcp_base @ T_correction)
        return homogeneous_to_tcp(T_tcp_base)

    # "Off (자동)": 매칭된 객체 자세를 그대로 TCP 자세로 사용 (회전 보정 없음).
    # 무작위 자세 시나리오에서 미리 정한 잡기 축이 의미 없을 때 사용.
    # 그리퍼는 객체의 X/Y/Z 축에 직접 정렬됨 → 사용자가 매칭 결과 보고 적절성 판단.
    if grasp_axis is None or str(grasp_axis).startswith("Off"):
        return _apply_tool_rotation(T_grasp_base)

    # 3. Tool 자세 결정: Tool +Z를 grasp_axis (또는 그 반대)에 정렬
    R_obj = T_grasp_base[:3, :3]
    axis_vec = {"X": R_obj[:, 0], "Y": R_obj[:, 1], "Z": R_obj[:, 2]}[grasp_axis]
    if grasp_flip:
        new_z = -axis_vec  # 위에서 아래로 = 객체 -axis
    else:
        new_z = axis_vec

    new_z = new_z / np.linalg.norm(new_z)

    # Tool X축은 객체의 다른 축 중 grasp_axis와 다른 것을 투영해서 사용
    # (grasp_axis가 Z면 객체 X축, grasp_axis가 X면 객체 Y축, ...)
    ref_axis_name = "X" if grasp_axis != "X" else "Y"
    ref_axis = {"X": R_obj[:, 0], "Y": R_obj[:, 1], "Z": R_obj[:, 2]}[ref_axis_name]
    new_x = ref_axis - np.dot(ref_axis, new_z) * new_z
    nx_norm = np.linalg.norm(new_x)
    if nx_norm < 1e-6:
        # fallback: 월드 X
        wx = np.array([1.0, 0.0, 0.0])
        new_x = wx - np.dot(wx, new_z) * new_z
        nx_norm = np.linalg.norm(new_x)
    new_x = new_x / nx_norm
    new_y = np.cross(new_z, new_x)

    R_tcp = np.column_stack([new_x, new_y, new_z])
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = R_tcp
    T_tcp[:3, 3] = T_grasp_base[:3, 3]

    return _apply_tool_rotation(T_tcp)


def _frame_from_z(z: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Tool +Z 방향과 참조 X 후보로부터 직교 3x3 회전(열=X/Y/Z축) 생성.

    compute_grasp_pose 3단계와 동일한 Gram-Schmidt 방식 — ref 를 z에 수직인
    평면에 투영해 X축으로 쓰고, 퇴화 시 월드 X→Y 순으로 fallback.
    """
    z = z / np.linalg.norm(z)
    x = ref - np.dot(ref, z) * z
    if np.linalg.norm(x) < 1e-6:
        wx = np.array([1.0, 0.0, 0.0])
        x = wx - np.dot(wx, z) * z
        if np.linalg.norm(x) < 1e-6:
            wy = np.array([0.0, 1.0, 0.0])
            x = wy - np.dot(wy, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _aligned_tool_frame(grasp_axis: str, grasp_flip: bool) -> Optional[np.ndarray]:
    """compute_grasp_pose 3단계의 축 정렬을 CAD 좌표계(객체 자세=I)에서 재현.

    반환: 3x3 회전 (열 = 정렬된 Tool X/Y/Z 축, CAD 좌표계 기준).
    잡기 축이 "Off (자동)"면 정렬 기준이 없으므로 None.
    """
    if grasp_axis is None or str(grasp_axis).startswith("Off"):
        return None
    eye = np.eye(3)
    axis_vec = {"X": eye[:, 0], "Y": eye[:, 1], "Z": eye[:, 2]}[grasp_axis]
    base_z = -axis_vec if grasp_flip else axis_vec
    ref_name = "X" if grasp_axis != "X" else "Y"
    ref_axis = {"X": eye[:, 0], "Y": eye[:, 1], "Z": eye[:, 2]}[ref_name]
    return _frame_from_z(base_z, ref_axis)


def tool_frame_from_normal(normal: np.ndarray, grasp_axis: str, grasp_flip: bool) -> Optional[np.ndarray]:
    """표면 법선 → 제안되는 최종 Tool 좌표계 (CAD 기준, 열 = X/Y/Z축).

    Tool+Z 는 표면 안쪽(-normal), X축은 정렬 X를 유지하려 시도해 손목
    비틀림(twist)을 최소화. 잡기 축 Off 또는 법선 퇴화 시 None.
    """
    R_aligned = _aligned_tool_frame(grasp_axis, grasp_flip)
    if R_aligned is None:
        return None
    n = np.asarray(normal, dtype=float)
    ln = np.linalg.norm(n)
    if not np.isfinite(ln) or ln < 1e-9:
        return None
    return _frame_from_z(-n / ln, R_aligned[:, 0])


def suggest_rotation_from_normal(normal: np.ndarray, grasp_axis: str, grasp_flip: bool) -> Optional[Tuple[float, float, float]]:
    """CAD 표면 법선 → 그리퍼가 그 면에 수직 접근하도록 grasp 회전(A/B/C deg) 제안.

    compute_grasp_pose 는 먼저 Tool+Z 를 잡기 축에 정렬한 뒤 Tool 좌표계 기준
    추가 회전(grasp_rotation_abc)을 곱한다. 여기서는 그 "추가 회전"을 역산:
    정렬 결과 R_aligned 와 원하는 자세(tool_frame_from_normal)의 차이를
    ZYX intrinsic (KUKA A/B/C)로 분해. 잡기 축 "Off (자동)"면 None.
    """
    R_aligned = _aligned_tool_frame(grasp_axis, grasp_flip)
    R_desired = tool_frame_from_normal(normal, grasp_axis, grasp_flip)
    if R_aligned is None or R_desired is None:
        return None
    R_add = R_aligned.T @ R_desired
    a, b, c = Rotation.from_matrix(R_add).as_euler("ZYX", degrees=True)
    return float(a), float(b), float(c)
