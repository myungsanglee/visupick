"""
CAD 매칭 탭 (6D pose estimation by ICP)
- STL/OBJ/PLY CAD 모델 로드
- 캡처 → 2D ROI 드래그 → ROI 내 포인트 클라우드 추출
- Open3D Global Registration (FPFH+RANSAC) + ICP 멀티 인스턴스 매칭
- 매칭된 6D 자세를 3D 뷰에 메시 오버레이로 표시
- 객체 선택 → 카메라→로봇 base 변환 → 이동 명령
"""

import copy
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QApplication,
    QComboBox,
    QSplitter,
    QCheckBox,
    QScrollArea,
    QProgressBar,
    QListWidget,
)

import cv2
import pyvista as pv
import open3d as o3d
from scipy.spatial.transform import Rotation

from calibration import tcp_to_homogeneous, homogeneous_to_tcp
from kuka_robot import normalize_robot_mode, is_auto_mode
from bin_picking_tab import DraggableImageLabel, PointCloudView3D
from robot_control_mixin import RobotControlMixin

# PPF 모듈은 opencv-contrib-python에만 있음. 없으면 PPF 옵션 자동 비활성.
HAS_PPF = hasattr(cv2, "ppf_match_3d")

logger = logging.getLogger(__name__)


# 객체/클러스터 공용 색상 팔레트 (3D 시각화 + 2D 라벨 매칭용)
INSTANCE_COLORS_RGB = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 140, 255),
    (255, 200, 60),
    (220, 100, 220),
    (80, 220, 220),
    (255, 140, 60),
    (200, 200, 200),
    (180, 60, 180),
    (60, 180, 180),
    (180, 180, 60),
    (140, 140, 255),
]


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
            maximum_tuple_count=5000,    # 더 많은 tuple 후보
            tuple_scale=0.85,            # 더 관대한 tuple test (기본 0.95)
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
            debug_log.append(
                f"[시도 {i + 1}] scene_down={len(scene_down.points)}점, FGR fitness={best_ransac.fitness:.3f}"
            )
        else:
            ransac_results = []
            for k in range(ransac_attempts):
                r = global_register(model_down, scene_down, model_fpfh, scene_fpfh, voxel_size)
                ransac_results.append(r)
            best_ransac = max(ransac_results, key=lambda r: r.fitness)
            fit_strs = ", ".join(f"{r.fitness:.3f}" for r in ransac_results)
            debug_log.append(
                f"[시도 {i + 1}] scene_down={len(scene_down.points)}점, RANSAC[{fit_strs}] → best={best_ransac.fitness:.3f}"
            )

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
    debug_log.append(
        f"DBSCAN: scene {len(scene_pcd.points)}점 → 클러스터 {len(clusters)}개 (eps={eps}mm, min_pts={min_points})"
    )
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
                f"[클러스터 {ci + 1}] {len(cluster.points)}점, scene_down={len(scene_down.points)}점, "
                f"FGR fitness={best_ransac.fitness:.3f}"
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
        debug_log.append(
            f"[클러스터 {ci + 1}] {len(scene_data)}점, PPF 후보 {len(results)}개, "
            f"상위 {n_use}개 votes={votes_before}"
        )

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
                    "fitness": cand["fitness"],   # Open3D fitness (0~1, 클수록 좋음)
                    "rmse": cand["rmse"],          # mm
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


# ============================================================
# CAD 매칭 탭
# ============================================================


class CADMatchingTab(RobotControlMixin, QWidget):
    """
    CAD 기반 6D pose estimation 탭

    흐름:
      1. STL/OBJ/PLY CAD 모델 로드
      2. 캘리브레이션 JSON 로드
      3. 캡처 → 2D 이미지 + 포인트 클라우드 표시
      4. 2D 뷰에서 ROI 드래그
      5. "매칭 실행" → 멀티 인스턴스 ICP 매칭
      6. 결과 테이블에서 객체 선택 → 6D → TCP 자세
      7. 이동 (Approach/Target/Retract)
    """

    DEFAULT_VOXEL_SIZE = 5.0  # mm
    DEFAULT_MAX_INSTANCES = 5
    DEFAULT_FITNESS_THRESHOLD = 0.20

    # 시퀀스 라벨/메시지에서 사용하는 대상 명사 (RobotControlMixin 오버라이드)
    SEQ_OBJECT_NOUN = "인스턴스"

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        # 입력 데이터
        self.current_image = None  # BGR
        self.current_xyz = None  # (H, W, 3) mm
        self.current_rgb = None  # (H, W, 3) uint8
        self.current_intrinsics = None  # 3x3
        self.roi_2d = None  # (x1, y1, x2, y2) 픽셀

        # CAD
        self.cad_path = None
        self.cad_pcd = None  # Open3D PointCloud (샘플링됨)
        self.cad_mesh = None  # Open3D TriangleMesh (시각화용, 옵셔널)
        self.cad_size = 0.0  # mm (대각선 길이)

        # PPF detector 캐시 (CAD 한 번 학습 후 매칭마다 재사용)
        self._ppf_detector = None
        self._ppf_model_data = None
        self._ppf_model_o3d = None  # Open3D ICP 정밀화용 (다운샘플 + normal 정렬된 model)
        self._ppf_cad_path = None  # 어떤 CAD 경로로 학습됐는지 추적

        # 알고리즘 전환 시 잡기 축 자동 복원용 (PPF↔FPFH 경계를 넘을 때만 토글)
        self._last_non_off_grasp_axis = "Z"
        self._last_cull_state = True
        self._prev_algo = "FPFH+ICP (RANSAC)"  # 직전 알고리즘 (PPF 경계 판정용)
        self._programmatic_axis_change = False  # 프로그램이 grasp_axis 바꾸는 중 재귀 가드
        self._flip_applied = False  # 현재 선택 인스턴스에 180° flip이 적용됐는지 (영속)

        # 3D 뷰에 추가한 actor 이름 추적 (range(20) 무차별 remove 대신 정확히 제거)
        self._instance_actor_names = []
        self._grasp_marker_names = []
        self._tcp_viz_actors = []  # 선택된 인스턴스의 TCP Tool 자세 시각화

        # Grasp 포인트 (CAD 좌표계 기준 mm). 매칭 후 객체에서 실제로 잡을 위치.
        # (0,0,0)이면 CAD 원점이 그립 위치. CAD마다 원점이 다르므로 사용자가 지정.
        self.grasp_position_cad = np.array([0.0, 0.0, 0.0])
        # Grasp 회전 (Tool 좌표계 기준 deg, KUKA ZYX intrinsic). 잡기 축 정렬 후 추가 보정.
        # (0,0,0)이면 보정 없음. A=yaw, B=pitch, C=roll.
        self.grasp_rotation_abc_deg = np.array([0.0, 0.0, 0.0])

        # 시퀀스 큐 (Python 측 사용자 큐). 각 액션은 {"type": "object_move"|"home", "label", "target", ...}
        self.user_queue = []

        # 캘리브레이션
        self.T_calib = None
        self.calib_mode = None

        # 매칭 결과
        self.instances: List[Dict] = []  # [{transformation, fitness, rmse}]
        self.selected_idx = None
        self.target_pose = None

        # 로봇 모드 캐시
        self._current_mode = "?"

        self._init_ui()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === 상단 1행: 데이터 로드 + 캡처 ===
        top1 = QHBoxLayout()

        self.btn_load_cad = QPushButton("CAD 로드 (STL/OBJ/PLY)")
        self.btn_load_cad.clicked.connect(self._load_cad)
        top1.addWidget(self.btn_load_cad)
        self.cad_label = QLabel("미로드")
        self.cad_label.setStyleSheet("color: #888;")
        top1.addWidget(self.cad_label)
        top1.addSpacing(15)

        self.btn_load_calib = QPushButton("캘리브레이션 (JSON)")
        self.btn_load_calib.clicked.connect(self._load_calibration)
        top1.addWidget(self.btn_load_calib)
        self.calib_label = QLabel("미로드")
        self.calib_label.setStyleSheet("color: #888;")
        top1.addWidget(self.calib_label)
        top1.addSpacing(15)

        self.btn_capture = QPushButton("캡처")
        self.btn_capture.clicked.connect(self._capture)
        top1.addWidget(self.btn_capture)

        self.btn_clear_roi = QPushButton("ROI 해제")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        top1.addWidget(self.btn_clear_roi)

        self.btn_preview_cad = QPushButton("CAD 미리보기")
        self.btn_preview_cad.clicked.connect(self._preview_cad)
        self.btn_preview_cad.setToolTip("3D 뷰에 CAD를 원점에 띄움. +Z(파랑)가 윗면 방향인지 확인용. cull 결과(빨강)도 함께 표시.")
        top1.addWidget(self.btn_preview_cad)

        top1.addWidget(QLabel("알고리즘:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("FPFH+ICP (RANSAC)")
        self.algo_combo.addItem("FPFH+ICP (FGR)")
        if HAS_PPF:
            self.algo_combo.addItem("PPF (OpenCV)")
        self.algo_combo.setToolTip(
            "FPFH+ICP (RANSAC): 무작위 샘플링 기반. 비결정적 (N회 시도해 best 채택).\n"
            "FPFH+ICP (FGR): Fast Global Registration. 결정적, RANSAC보다 빠름. 같은 입력 → 항상 같은 결과.\n"
            "PPF: 무작위 자세에 강건. DBSCAN과 함께 쓰면 빈 픽킹 표준 구성. 첫 매칭은 학습 시간 포함."
        )
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        top1.addWidget(self.algo_combo)

        self.btn_match = QPushButton("매칭 실행")
        self.btn_match.clicked.connect(self._run_matching)
        self.btn_match.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        top1.addWidget(self.btn_match)

        top1.addStretch()

        # 로봇 모드 표시
        self.mode_label = QLabel("모드: ?")
        self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; background-color: #BDBDBD; color: white; border-radius: 3px;")
        top1.addWidget(self.mode_label)

        # 뷰 스위치
        self.btn_view_2d = QPushButton("2D 뷰")
        self.btn_view_2d.setCheckable(True)
        self.btn_view_2d.setChecked(True)
        self.btn_view_2d.clicked.connect(lambda: self._switch_view(0))
        top1.addWidget(self.btn_view_2d)
        self.btn_view_3d = QPushButton("3D 뷰")
        self.btn_view_3d.setCheckable(True)
        self.btn_view_3d.clicked.connect(lambda: self._switch_view(1))
        top1.addWidget(self.btn_view_3d)
        self.btn_view_cad = QPushButton("CAD 뷰")
        self.btn_view_cad.setCheckable(True)
        self.btn_view_cad.clicked.connect(lambda: self._switch_view(2))
        self.btn_view_cad.setToolTip("CAD 미리보기 전용 3D 뷰. Zivid 포인트클라우드 뷰와 분리됨.")
        top1.addWidget(self.btn_view_cad)
        self.btn_view_cluster = QPushButton("클러스터 뷰")
        self.btn_view_cluster.setCheckable(True)
        self.btn_view_cluster.clicked.connect(lambda: self._switch_view(3))
        self.btn_view_cluster.setToolTip("DBSCAN 클러스터 미리보기 전용 3D 뷰. 다른 뷰와 분리됨.")
        top1.addWidget(self.btn_view_cluster)

        layout.addLayout(top1)

        # === 상단 2행: 매칭 파라미터 ===
        top2 = QHBoxLayout()
        top2.addWidget(QLabel("Voxel(mm):"))
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(0.5, 50.0)
        self.voxel_spin.setSingleStep(0.5)
        self.voxel_spin.setValue(self.DEFAULT_VOXEL_SIZE)
        self.voxel_spin.setFixedWidth(80)
        self.voxel_spin.setToolTip("다운샘플 voxel 크기. 객체 크기의 1/30 ~ 1/50 권장.")
        top2.addWidget(self.voxel_spin)

        top2.addWidget(QLabel("최대 인스턴스:"))
        self.max_inst_spin = QSpinBox()
        self.max_inst_spin.setRange(1, 20)
        self.max_inst_spin.setValue(self.DEFAULT_MAX_INSTANCES)
        self.max_inst_spin.setFixedWidth(60)
        top2.addWidget(self.max_inst_spin)

        top2.addWidget(QLabel("Fitness 임계값:"))
        self.fitness_spin = QDoubleSpinBox()
        self.fitness_spin.setRange(0.05, 1.0)
        self.fitness_spin.setSingleStep(0.05)
        self.fitness_spin.setValue(self.DEFAULT_FITNESS_THRESHOLD)
        self.fitness_spin.setFixedWidth(80)
        self.fitness_spin.setToolTip("ICP fitness가 이보다 낮으면 매칭 실패로 간주. 0.3~0.5 권장.")
        top2.addWidget(self.fitness_spin)

        top2.addWidget(QLabel("RANSAC 시도:"))
        self.ransac_attempts_spin = QSpinBox()
        self.ransac_attempts_spin.setRange(1, 20)
        self.ransac_attempts_spin.setValue(5)
        self.ransac_attempts_spin.setFixedWidth(60)
        self.ransac_attempts_spin.setToolTip(
            "각 인스턴스 검색 시 RANSAC을 N번 반복해 가장 fitness 높은 결과 채택.\n"
            "클수록 안정적이지만 시간이 N배 늘어남. 시드 고정으로 같은 입력엔 항상 같은 결과."
        )
        top2.addWidget(self.ransac_attempts_spin)

        top2.addSpacing(15)
        top2.addWidget(QLabel("잡기 축:"))
        self.grasp_axis_combo = QComboBox()
        self.grasp_axis_combo.addItems(["Z", "X", "Y", "Off (자동)"])
        self.grasp_axis_combo.setToolTip(
            "CAD 좌표계의 어느 축을 그리퍼 +Z 방향에 정렬할지.\n"
            "Off: 매칭된 객체 자세를 그대로 TCP 자세로 사용 (무작위 자세 시나리오용).\n"
            "Off 선택 시 'CAD 보이는 면만 사용' 옵션도 자동 비활성."
        )
        self.grasp_axis_combo.currentTextChanged.connect(self._on_grasp_axis_changed)
        top2.addWidget(self.grasp_axis_combo)

        self.grasp_flip_check = QCheckBox("축 뒤집기 (Tool+Z = -axis)")
        self.grasp_flip_check.setChecked(True)
        self.grasp_flip_check.setToolTip("True: 객체 위에서 아래로 잡음 (대부분의 빈 픽킹). 잡기 축이 Off일 때는 무시됨.")
        top2.addWidget(self.grasp_flip_check)

        top2.addSpacing(15)
        self.cull_visible_check = QCheckBox("CAD 보이는 면만 사용")
        self.cull_visible_check.setChecked(True)
        self.cull_visible_check.setToolTip(
            "카메라 시점에서 보이는 면(잡기 축 + 방향)의 점들만 매칭에 사용.\n"
            "위에서 아래로 촬영하는 정자세 환경에서는 거의 필수.\n"
            "무작위 자세(빈 픽킹)에서는 끄는 게 맞음. 잡기 축이 Off면 자동 비활성."
        )
        top2.addWidget(self.cull_visible_check)

        top2.addStretch()
        layout.addLayout(top2)

        # === 상단 3행: DBSCAN 클러스터링 옵션 ===
        top3 = QHBoxLayout()
        self.use_dbscan = QCheckBox("DBSCAN 클러스터별 매칭")
        self.use_dbscan.setChecked(False)
        self.use_dbscan.setToolTip(
            "ROI 안의 포인트 클라우드를 객체 단위로 자동 분리한 뒤, 각 클러스터마다 한 번씩 매칭.\n"
            "장점: 멀티 인스턴스 + 무작위 자세 시나리오에 강함. RANSAC이 두 객체 사이를 매칭하는 사고 방지.\n"
            "한계: 맞닿거나 겹친 객체는 한 클러스터로 묶여 분리 실패."
        )
        top3.addWidget(self.use_dbscan)

        top3.addWidget(QLabel("eps(mm):"))
        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setRange(1.0, 100.0)
        self.dbscan_eps.setValue(15.0)
        self.dbscan_eps.setSingleStep(1.0)
        self.dbscan_eps.setFixedWidth(70)
        self.dbscan_eps.setToolTip("같은 클러스터로 묶일 점 사이 최대 거리. 작은 객체나 빽빽한 환경엔 작게 (5~10), 떨어진 환경엔 크게 (15~30).")
        top3.addWidget(self.dbscan_eps)

        top3.addWidget(QLabel("min pts:"))
        self.dbscan_min_pts = QSpinBox()
        self.dbscan_min_pts.setRange(5, 5000)
        self.dbscan_min_pts.setValue(100)
        self.dbscan_min_pts.setFixedWidth(80)
        self.dbscan_min_pts.setToolTip("클러스터로 인정될 최소 점 수. 너무 작으면 노이즈가 클러스터됨.")
        top3.addWidget(self.dbscan_min_pts)

        self.btn_preview_clusters = QPushButton("클러스터 미리보기")
        self.btn_preview_clusters.setToolTip("매칭 안 하고 DBSCAN 결과만 클러스터 뷰에 색깔별로 표시 (eps/min_pts 조정용).")
        self.btn_preview_clusters.clicked.connect(self._preview_clusters)
        top3.addWidget(self.btn_preview_clusters)

        top3.addSpacing(15)
        self.remove_plane_check = QCheckBox("작업대 평면 제거")
        self.remove_plane_check.setChecked(False)
        self.remove_plane_check.setToolTip(
            "RANSAC으로 가장 큰 평면(작업대 표면)을 검출해 그 점들을 제거.\n"
            "작업대 점이 한 거대 클러스터를 만들어 객체가 묻히는 걸 방지.\n"
            "객체 윗면이 평평하고 클 경우 객체까지 같이 사라질 수 있으니 주의."
        )
        top3.addWidget(self.remove_plane_check)

        top3.addWidget(QLabel("평면 두께(mm):"))
        self.plane_dist_spin = QDoubleSpinBox()
        self.plane_dist_spin.setRange(0.5, 30.0)
        self.plane_dist_spin.setValue(5.0)
        self.plane_dist_spin.setSingleStep(0.5)
        self.plane_dist_spin.setFixedWidth(70)
        self.plane_dist_spin.setToolTip("평면에 속한다고 인정할 최대 거리. 작업대 평탄도와 Zivid 노이즈에 따라 2~10mm.")
        top3.addWidget(self.plane_dist_spin)

        top3.addStretch()
        layout.addLayout(top3)

        # === 상단 4행: Grasp 포인트 설정 (CAD 좌표계 기준) ===
        top4 = QHBoxLayout()
        top4.addWidget(QLabel("Grasp 위치 (CAD mm):"))
        top4.addWidget(QLabel("X"))
        self.grasp_x_spin = QDoubleSpinBox()
        self.grasp_x_spin.setRange(-1000.0, 1000.0)
        self.grasp_x_spin.setSingleStep(1.0)
        self.grasp_x_spin.setDecimals(1)
        self.grasp_x_spin.setFixedWidth(80)
        self.grasp_x_spin.valueChanged.connect(self._on_grasp_position_changed)
        top4.addWidget(self.grasp_x_spin)
        top4.addWidget(QLabel("Y"))
        self.grasp_y_spin = QDoubleSpinBox()
        self.grasp_y_spin.setRange(-1000.0, 1000.0)
        self.grasp_y_spin.setSingleStep(1.0)
        self.grasp_y_spin.setDecimals(1)
        self.grasp_y_spin.setFixedWidth(80)
        self.grasp_y_spin.valueChanged.connect(self._on_grasp_position_changed)
        top4.addWidget(self.grasp_y_spin)
        top4.addWidget(QLabel("Z"))
        self.grasp_z_spin = QDoubleSpinBox()
        self.grasp_z_spin.setRange(-1000.0, 1000.0)
        self.grasp_z_spin.setSingleStep(1.0)
        self.grasp_z_spin.setDecimals(1)
        self.grasp_z_spin.setFixedWidth(80)
        self.grasp_z_spin.valueChanged.connect(self._on_grasp_position_changed)
        top4.addWidget(self.grasp_z_spin)

        self.btn_grasp_reset = QPushButton("원점(0,0,0)")
        self.btn_grasp_reset.setToolTip("Grasp 위치를 CAD 좌표 원점(0,0,0)으로 리셋. CAD 원점이 곧 잡는 위치가 됨.")
        self.btn_grasp_reset.clicked.connect(self._grasp_to_origin)
        top4.addWidget(self.btn_grasp_reset)

        self.btn_grasp_center = QPushButton("객체 중심")
        self.btn_grasp_center.setToolTip("Grasp 위치를 CAD bounding box 중심으로 설정. CAD 원점이 객체 외부에 있을 때 유용.")
        self.btn_grasp_center.clicked.connect(self._grasp_to_center)
        top4.addWidget(self.btn_grasp_center)

        top4.addStretch()
        layout.addLayout(top4)

        # === 상단 5행: Grasp 회전 보정 (Tool 좌표계 기준, KUKA ZYX intrinsic) ===
        top5 = QHBoxLayout()
        top5.addWidget(QLabel("Grasp 회전 (Tool deg):"))
        top5.addWidget(QLabel("A"))
        self.grasp_a_spin = QDoubleSpinBox()
        self.grasp_a_spin.setRange(-180.0, 180.0)
        self.grasp_a_spin.setSingleStep(5.0)
        self.grasp_a_spin.setDecimals(1)
        self.grasp_a_spin.setFixedWidth(80)
        self.grasp_a_spin.setToolTip("Tool +Z 둘레 회전 (yaw, 그리퍼 손목 돌림). 평면 잡기 정렬에 가장 자주 쓰임.")
        self.grasp_a_spin.valueChanged.connect(self._on_grasp_rotation_changed)
        top5.addWidget(self.grasp_a_spin)
        top5.addWidget(QLabel("B"))
        self.grasp_b_spin = QDoubleSpinBox()
        self.grasp_b_spin.setRange(-180.0, 180.0)
        self.grasp_b_spin.setSingleStep(5.0)
        self.grasp_b_spin.setDecimals(1)
        self.grasp_b_spin.setFixedWidth(80)
        self.grasp_b_spin.setToolTip("Tool +Y 둘레 회전 (pitch, 비스듬한 접근).")
        self.grasp_b_spin.valueChanged.connect(self._on_grasp_rotation_changed)
        top5.addWidget(self.grasp_b_spin)
        top5.addWidget(QLabel("C"))
        self.grasp_c_spin = QDoubleSpinBox()
        self.grasp_c_spin.setRange(-180.0, 180.0)
        self.grasp_c_spin.setSingleStep(5.0)
        self.grasp_c_spin.setDecimals(1)
        self.grasp_c_spin.setFixedWidth(80)
        self.grasp_c_spin.setToolTip("Tool +X 둘레 회전 (roll, 옆으로 기울임).")
        self.grasp_c_spin.valueChanged.connect(self._on_grasp_rotation_changed)
        top5.addWidget(self.grasp_c_spin)

        self.btn_grasp_rot_reset = QPushButton("회전 0으로")
        self.btn_grasp_rot_reset.setToolTip("Grasp 회전을 (0, 0, 0)으로 리셋. 기본 잡기 축 정렬만 사용.")
        self.btn_grasp_rot_reset.clicked.connect(self._grasp_rotation_reset)
        top5.addWidget(self.btn_grasp_rot_reset)

        top5.addStretch()
        layout.addLayout(top5)

        # 진행 표시
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # === 중앙: 뷰 + 정보 ===
        splitter = QSplitter(Qt.Horizontal)

        # 좌: 2D/3D 스택
        self.view_stack = QStackedWidget()
        self.view_2d = DraggableImageLabel()
        self.view_2d.roiChanged.connect(self._on_roi_dragged)
        self.view_stack.addWidget(self.view_2d)

        self.view_3d = PointCloudView3D()
        self.view_stack.addWidget(self.view_3d)

        # CAD 미리보기 전용 3D 뷰 (view_3d와 독립적, 충돌 회피)
        self.view_cad = PointCloudView3D()
        self.view_stack.addWidget(self.view_cad)

        # 클러스터 미리보기 전용 3D 뷰 (DBSCAN 결과 시각화)
        self.view_cluster = PointCloudView3D()
        self.view_stack.addWidget(self.view_cluster)
        splitter.addWidget(self.view_stack)

        # 우: 정보 패널
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # 매칭 결과 테이블
        match_group = QGroupBox("매칭된 인스턴스")
        match_layout = QVBoxLayout(match_group)
        self.inst_table = QTableWidget(0, 4)
        self.inst_table.setHorizontalHeaderLabels(["#", "Fitness", "RMSE(mm)", "위치(camera)"])
        self.inst_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.inst_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.inst_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.inst_table.itemSelectionChanged.connect(self._on_table_selection)
        match_layout.addWidget(self.inst_table)
        info_layout.addWidget(match_group)

        # 선택된 객체 → TCP 자세
        sel_group = QGroupBox("선택된 객체의 TCP 자세 (로봇 base)")
        sel_layout = QVBoxLayout(sel_group)
        self.tcp_labels = {}
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis}:"))
            lbl = QLabel("---")
            lbl.setStyleSheet("font-family: monospace; font-size: 13px;")
            self.tcp_labels[axis] = lbl
            row.addWidget(lbl)
            row.addStretch()
            sel_layout.addLayout(row)

        # 180° 회전 보정 (대칭 객체에서 매칭 결과가 180° 어긋날 때 수동 보정)
        self.btn_flip_180 = QPushButton("180° 회전 (Tool +Z축 둘레)")
        self.btn_flip_180.setToolTip(
            "차단기처럼 윗면이 대칭인 객체는 매칭 결과가 180° 어긋날 수 있음.\n"
            "3D 뷰에서 ON/OFF 스위치 위치가 반대로 잡혔으면 이 버튼으로 한 번에 보정.\n"
            "Tool +Z축 둘레로 회전 → 잡는 위치는 그대로, 그리퍼 방향만 반대."
        )
        self.btn_flip_180.clicked.connect(self._flip_target_180)
        sel_layout.addWidget(self.btn_flip_180)
        info_layout.addWidget(sel_group)

        # ============================================================
        # 로봇 이동 제어 (bin_picking_tab과 동일 구성)
        # ============================================================
        move_group = QGroupBox("로봇 이동 제어")
        move_layout = QVBoxLayout(move_group)

        # 이동 방식
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("방식:"))
        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(["LIN (직선, 추천)", "PTP (최단 경로)"])
        opt_row.addWidget(self.move_mode_combo)
        move_layout.addLayout(opt_row)

        # 속도 + 적용 버튼
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("속도(%):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(30)
        self.speed_spin.setFixedWidth(70)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_spin)
        self.btn_apply_speed = QPushButton("적용")
        self.btn_apply_speed.setFixedWidth(50)
        self.btn_apply_speed.clicked.connect(self._apply_speed_now)
        speed_row.addWidget(self.btn_apply_speed)
        speed_row.addStretch()
        move_layout.addLayout(speed_row)

        # Approach / Retract
        approach_row = QHBoxLayout()
        self.use_approach = QCheckBox("접근/철수 사용")
        self.use_approach.setChecked(True)
        self.use_approach.setToolTip(
            "체크 시 [Approach → Target → Retract] 3단계 모션을 큐에 추가\n"
            "Tool +Z 방향으로 위에 안전하게 다가갔다 → 정밀 접근 → 다시 위로"
        )
        approach_row.addWidget(self.use_approach)
        approach_row.addWidget(QLabel("거리(mm):"))
        self.approach_dist = QSpinBox()
        self.approach_dist.setRange(5, 500)
        self.approach_dist.setValue(50)
        self.approach_dist.setFixedWidth(60)
        approach_row.addWidget(self.approach_dist)
        approach_row.addStretch()
        move_layout.addLayout(approach_row)

        # Z 최소 한계
        zlim_row = QHBoxLayout()
        zlim_row.addWidget(QLabel("Z 최소(mm):"))
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(-2000, 2000)
        self.z_min_spin.setValue(5)
        self.z_min_spin.setFixedWidth(80)
        self.z_min_spin.setToolTip("타겟 Z 좌표가 이 값보다 낮으면 이동을 거부합니다 (바닥 충돌 방지)")
        zlim_row.addWidget(self.z_min_spin)
        zlim_row.addStretch()
        move_layout.addLayout(zlim_row)

        # 이동 버튼 (큰 파랑)
        self.btn_move = QPushButton("선택 위치로 이동")
        self.btn_move.setMinimumHeight(45)
        self.btn_move.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1976D2; color: white;")
        self.btn_move.clicked.connect(self._execute_move)
        self.btn_move.setEnabled(False)
        move_layout.addWidget(self.btn_move)

        # Home 이동/재설정 (한 줄)
        home_row = QHBoxLayout()
        self.btn_move_home = QPushButton("🏠 Home으로 이동")
        self.btn_move_home.setMinimumHeight(40)
        self.btn_move_home.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #2E7D32; color: white;")
        self.btn_move_home.clicked.connect(self._move_to_home)
        self.btn_move_home.setEnabled(False)
        home_row.addWidget(self.btn_move_home, stretch=2)

        self.btn_set_home = QPushButton("📍 Home\n재설정")
        self.btn_set_home.setMinimumHeight(40)
        self.btn_set_home.setStyleSheet("font-size: 11px; background-color: #689F38; color: white;")
        self.btn_set_home.setToolTip("현재 로봇 TCP 위치를 새 Home으로 저장합니다")
        self.btn_set_home.clicked.connect(self._set_home_to_current)
        self.btn_set_home.setEnabled(False)
        home_row.addWidget(self.btn_set_home, stretch=1)
        move_layout.addLayout(home_row)

        # 큐 비우기
        self.btn_clear_queue = QPushButton("🗑 큐 비우기 (이전 명령 취소)")
        self.btn_clear_queue.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold;")
        self.btn_clear_queue.clicked.connect(self._clear_motion_queue)
        move_layout.addWidget(self.btn_clear_queue)

        # 비상정지 (큰 빨강)
        self.btn_estop = QPushButton("⛔ 비상정지 (Space)")
        self.btn_estop.setMinimumHeight(60)
        self.btn_estop.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #D32F2F; color: white;")
        self.btn_estop.clicked.connect(self._emergency_stop)
        move_layout.addWidget(self.btn_estop)

        # 비상정지 해제 (작게)
        self.btn_estop_release = QPushButton("비상정지 해제")
        self.btn_estop_release.setStyleSheet("background-color: #757575; color: white;")
        self.btn_estop_release.clicked.connect(self._emergency_stop_release)
        move_layout.addWidget(self.btn_estop_release)

        info_layout.addWidget(move_group)

        # ============================================================
        # 시퀀스 큐 (자동 실행 시나리오)
        # ============================================================
        seq_group = QGroupBox("시퀀스 큐 (자동 실행 순서)")
        seq_layout = QVBoxLayout(seq_group)

        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(80)
        self.action_list.setMaximumHeight(150)
        seq_layout.addWidget(self.action_list)

        # 추가 버튼들
        add_row = QHBoxLayout()
        self.btn_add_obj_to_seq = QPushButton("➕ 객체 이동 추가")
        self.btn_add_obj_to_seq.setStyleSheet("background-color: #1976D2; color: white;")
        self.btn_add_obj_to_seq.clicked.connect(self._enqueue_object_move)
        self.btn_add_obj_to_seq.setEnabled(False)
        add_row.addWidget(self.btn_add_obj_to_seq)

        self.btn_add_home_to_seq = QPushButton("➕ Home 추가")
        self.btn_add_home_to_seq.setStyleSheet("background-color: #2E7D32; color: white;")
        self.btn_add_home_to_seq.clicked.connect(self._enqueue_home_to_sequence)
        self.btn_add_home_to_seq.setEnabled(False)
        add_row.addWidget(self.btn_add_home_to_seq)
        seq_layout.addLayout(add_row)

        # 제거 버튼들
        del_row = QHBoxLayout()
        self.btn_remove_seq_item = QPushButton("선택 항목 제거")
        self.btn_remove_seq_item.clicked.connect(self._remove_selected_action)
        del_row.addWidget(self.btn_remove_seq_item)

        self.btn_clear_seq = QPushButton("시퀀스 비우기")
        self.btn_clear_seq.clicked.connect(self._clear_user_queue)
        del_row.addWidget(self.btn_clear_seq)
        seq_layout.addLayout(del_row)

        # 시작 버튼 (큰 파랑)
        self.btn_start_seq = QPushButton("▶ 시퀀스 시작")
        self.btn_start_seq.setMinimumHeight(45)
        self.btn_start_seq.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #1565C0; color: white;")
        self.btn_start_seq.clicked.connect(self._start_sequence)
        seq_layout.addWidget(self.btn_start_seq)

        info_layout.addWidget(seq_group)

        info_layout.addStretch()

        # 스크롤 처리
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(info_widget)
        scroll.setMinimumWidth(380)
        splitter.addWidget(scroll)

        splitter.setSizes([900, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 스페이스바 = 비상정지 (탭이 활성일 때만 동작)
        from PySide6.QtGui import QShortcut, QKeySequence
        sc_estop = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_estop.setContext(Qt.WidgetWithChildrenShortcut)
        sc_estop.activated.connect(self._emergency_stop)

        # 모드 폴링 타이머
        self._mode_timer = QTimer(self)
        self._mode_timer.timeout.connect(self._refresh_mode_display)
        self._mode_timer.start(2000)

    def _on_grasp_position_changed(self):
        """Grasp spin 값 변경 시 CAD 미리보기와 현재 선택된 인스턴스 TCP 자세 모두 갱신."""
        self.grasp_position_cad = np.array([
            float(self.grasp_x_spin.value()),
            float(self.grasp_y_spin.value()),
            float(self.grasp_z_spin.value()),
        ])
        # CAD 뷰가 열려 있으면 마커 위치 갱신 (전체 재그림은 비싸니까 마커만)
        if self.cad_pcd is not None:
            self._update_grasp_marker_in_cad_view()
        # 매칭된 인스턴스가 선택되어 있으면 TCP 자세 재계산 (조용히 - 모달 억제)
        if self.selected_idx is not None:
            self._select_instance(self.selected_idx, silent=True)
        # 3D 매칭 결과 뷰의 grasp 마커도 갱신
        if self.instances:
            self._update_grasp_markers_in_3d_view()

    def _on_grasp_rotation_changed(self):
        """Grasp 회전 spin 변경 시 현재 선택된 인스턴스 TCP 자세 재계산."""
        self.grasp_rotation_abc_deg = np.array([
            float(self.grasp_a_spin.value()),
            float(self.grasp_b_spin.value()),
            float(self.grasp_c_spin.value()),
        ])
        if self.selected_idx is not None:
            self._select_instance(self.selected_idx, silent=True)

    def _grasp_rotation_reset(self):
        self.grasp_a_spin.setValue(0.0)
        self.grasp_b_spin.setValue(0.0)
        self.grasp_c_spin.setValue(0.0)

    def _grasp_to_origin(self):
        self.grasp_x_spin.setValue(0.0)
        self.grasp_y_spin.setValue(0.0)
        self.grasp_z_spin.setValue(0.0)

    def _grasp_to_center(self):
        """CAD bounding box 중심을 grasp 위치로 설정."""
        if self.cad_pcd is None:
            QMessageBox.warning(self, "오류", "CAD를 먼저 로드하세요")
            return
        pts = np.asarray(self.cad_pcd.points)
        if len(pts) == 0:
            return
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        self.grasp_x_spin.setValue(float(center[0]))
        self.grasp_y_spin.setValue(float(center[1]))
        self.grasp_z_spin.setValue(float(center[2]))

    def _update_grasp_marker_in_cad_view(self):
        """CAD 미리보기 뷰의 grasp 마커만 갱신 (전체 재그림 없이)."""
        plotter = self.view_cad.plotter
        try:
            plotter.remove_actor("grasp_marker")
        except Exception:
            pass
        try:
            plotter.remove_actor("grasp_arrow")
        except Exception:
            pass
        # CAD 미리보기가 활성화된 상태일 때만 마커 추가 (cad_full 액터로 판단)
        if "cad_full" not in plotter.actors:
            return
        gx, gy, gz = self.grasp_position_cad
        sphere = pv.Sphere(radius=6, center=[gx, gy, gz])
        plotter.add_mesh(sphere, color="#22ff22", name="grasp_marker", pickable=False, reset_camera=False)
        # 잡기 축 방향 화살표 (잡기 축 + 방향)
        axis_text = self.grasp_axis_combo.currentText()
        if not axis_text.startswith("Off"):
            axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis_text]
            sign = -1.0 if self.grasp_flip_check.isChecked() else 1.0
            direction = np.zeros(3)
            direction[axis_idx] = sign * 40.0  # 길이 40mm
            arrow = pv.Arrow(start=[gx, gy, gz], direction=direction, scale=1.0)
            plotter.add_mesh(arrow, color="#22ff22", name="grasp_arrow", pickable=False, reset_camera=False)
        plotter.render()

    def _update_grasp_markers_in_3d_view(self):
        """매칭 결과 3D 뷰에 각 인스턴스의 grasp 위치 표시."""
        plotter = self.view_3d.plotter
        # 직전에 추가한 grasp 마커만 제거
        for name in self._grasp_marker_names:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._grasp_marker_names.clear()

        if not self.instances:
            plotter.render()
            return
        grasp_local = np.array([
            self.grasp_position_cad[0],
            self.grasp_position_cad[1],
            self.grasp_position_cad[2],
            1.0,
        ])
        for i, inst in enumerate(self.instances):
            T = inst["transformation"]
            color01 = tuple(c / 255.0 for c in INSTANCE_COLORS_RGB[i % len(INSTANCE_COLORS_RGB)])
            grasp_in_cam = (T @ grasp_local)[:3]
            sphere = pv.Sphere(radius=5, center=grasp_in_cam)
            plotter.add_mesh(sphere, color=color01, name=f"inst_grasp_{i}", pickable=False, reset_camera=False)
            self._grasp_marker_names.append(f"inst_grasp_{i}")
        plotter.render()

    def _set_grasp_axis_programmatically(self, axis_text_prefix: str):
        """grasp_axis 콤보를 프로그램이 변경 (재귀 가드 적용 → _last 추적 안 됨)."""
        self._programmatic_axis_change = True
        try:
            for i in range(self.grasp_axis_combo.count()):
                if self.grasp_axis_combo.itemText(i).startswith(axis_text_prefix):
                    self.grasp_axis_combo.setCurrentIndex(i)
                    break
        finally:
            self._programmatic_axis_change = False

    def _on_algo_changed(self, text: str):
        """
        알고리즘 전환 시 의존 옵션 자동 토글 — PPF↔FPFH 경계를 넘을 때만.

          FPFH → PPF: cull 선호 저장 후 cull OFF + DBSCAN ON. 잡기 축은 유지.
          PPF → FPFH: 저장해둔 cull 상태 복원. 잡기 축은 유지.
          같은 군 내 전환 (RANSAC↔FGR, PPF↔PPF): 아무것도 안 건드림.

        잡기 축은 PPF에서도 TCP 자세 계산에 필요(`object_pose_to_tcp`).
        CAD 좌표계의 윗면 축(예: +Y)을 미리 정해뒀으면 PPF로 가도 그대로 사용.
        cull만 OFF가 되어 매칭은 전체 model로, TCP 자세는 잡기 축 정렬 그대로.
        """
        prev_was_ppf = self._prev_algo.startswith("PPF")
        is_ppf = text.startswith("PPF")
        self._prev_algo = text

        if is_ppf and not prev_was_ppf:
            # FPFH → PPF 경계: cull 선호 저장 후 OFF + DBSCAN ON
            if self.cull_visible_check.isEnabled():
                self._last_cull_state = self.cull_visible_check.isChecked()
            self.cull_visible_check.setChecked(False)
            self.use_dbscan.setChecked(True)
            self.main.statusBar().showMessage(
                "PPF 모드: cull OFF + DBSCAN 자동 활성화. 잡기 축은 유지 (TCP 자세 계산용). "
                "첫 매칭은 학습 시간(수~수십초) 포함."
            )
        elif not is_ppf and prev_was_ppf:
            # PPF → FPFH 경계: cull 상태 복원
            if self.cull_visible_check.isEnabled():
                self.cull_visible_check.setChecked(self._last_cull_state)
            self.main.statusBar().showMessage(
                f"FPFH+ICP 모드: cull '{self._last_cull_state}' 복원."
            )
        else:
            self.main.statusBar().showMessage(f"{text} 모드.")

    def _on_grasp_axis_changed(self, text: str):
        """잡기 축이 'Off'면 cull 체크박스 비활성+해제. 사용자 수동 변경만 _last 추적."""
        is_off = text.startswith("Off")
        if is_off:
            self.cull_visible_check.setChecked(False)
            self.cull_visible_check.setEnabled(False)
            self.grasp_flip_check.setEnabled(False)
        else:
            self.cull_visible_check.setEnabled(True)
            self.grasp_flip_check.setEnabled(True)
            # 사용자가 직접 비-Off 축을 고른 경우만 마지막 선호로 기억
            # (프로그램이 _on_algo_changed에서 바꾼 경우는 재귀 가드로 제외)
            if not self._programmatic_axis_change:
                self._last_non_off_grasp_axis = text

    def _switch_view(self, idx: int):
        self.view_stack.setCurrentIndex(idx)
        self.btn_view_2d.setChecked(idx == 0)
        self.btn_view_3d.setChecked(idx == 1)
        self.btn_view_cad.setChecked(idx == 2)
        self.btn_view_cluster.setChecked(idx == 3)
        if idx == 1:
            self.view_3d.refresh_camera()
        elif idx == 2:
            self.view_cad.refresh_camera()
        elif idx == 3:
            self.view_cluster.refresh_camera()

    # ---------------------------------------------------------
    # 데이터 로드
    # ---------------------------------------------------------

    def _load_cad(self):
        # 프로젝트 루트의 cad_models/ 를 기본 시작 위치로 (절대 경로 하드코딩 제거)
        default_dir = str(Path(__file__).parent / "cad_models")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "CAD 모델 선택",
            default_dir,
            "3D Models (*.stl *.obj *.ply);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        self.main.statusBar().showMessage("CAD 로드 중...")
        QApplication.processEvents()

        result = load_cad_model(path, n_sample_points=8000)
        if result is None:
            QMessageBox.critical(self, "오류", f"CAD 로드 실패:\n{path}")
            return

        pcd, mesh = result
        self.cad_pcd = pcd
        self.cad_mesh = mesh
        self.cad_path = path

        # 새 CAD가 로드되면 기존 PPF 학습 결과 무효화
        self._ppf_detector = None
        self._ppf_model_data = None
        self._ppf_model_o3d = None
        self._ppf_cad_path = None

        # CAD 크기 (대각선)
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        self.cad_size = float(np.linalg.norm(extent))

        # 권장 voxel size 자동 추정 (대각선의 1/40)
        recommended_voxel = max(0.5, self.cad_size / 40.0)
        self.voxel_spin.setValue(round(recommended_voxel, 1))

        n_pts = len(pcd.points)
        name = Path(path).name
        self.cad_label.setText(f"{name} ({n_pts}점, ~{self.cad_size:.1f}mm)")
        self.cad_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.main.statusBar().showMessage(f"CAD 로드 완료: {name}, voxel size {recommended_voxel:.1f}mm 권장")
        logger.info(f"CAD 로드: {path}, {n_pts}점, 크기 {self.cad_size:.1f}mm")

    def _load_calibration(self):
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "캘리브레이션 결과 (JSON)", "data", "JSON (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        try:
            with open(path) as f:
                result = json.load(f)
            self.T_calib = np.array(result["transformation_matrix"])
            self.calib_mode = result.get("mode", "eye_to_hand")
            self.calib_label.setText(f"{Path(path).name} [{self.calib_mode}]")
            self.calib_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.main.statusBar().showMessage(f"캘리브레이션 로드: {self.calib_mode}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"로드 실패:\n{e}")

    # ---------------------------------------------------------
    # 캡처
    # ---------------------------------------------------------

    def _capture(self):
        if not self.main.camera or not self.main.camera.connected:
            QMessageBox.warning(self, "오류", "카메라가 연결되지 않았습니다")
            return
        if not self.main.camera.is_capture_ready:
            QMessageBox.warning(self, "오류", "카메라가 캡처 준비되지 않았습니다 (Zivid 는 YML 로드 필요)")
            return

        self.main.statusBar().showMessage("캡처 중...")
        QApplication.processEvents()

        frame = self.main.camera.capture()
        if frame is None:
            self.main.statusBar().showMessage("캡처 실패")
            return

        image = self.main.camera.frame_to_2d_image(frame)
        xyz = self.main.camera.frame_to_point_cloud(frame)
        if image is None or xyz is None:
            self.main.statusBar().showMessage("이미지/포인트클라우드 추출 실패")
            return

        self.current_image = image
        self.current_xyz = xyz
        # PyVista용 RGB
        import cv2

        self.current_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # intrinsics
        intr_data = self.main.camera.get_intrinsics()
        if intr_data:
            self.current_intrinsics = np.array(intr_data["camera_matrix"])

        # 2D 뷰 갱신
        self.view_2d.set_image(image)
        # 3D 뷰 갱신
        self.view_3d.clear()
        self.view_3d.show_pointcloud(self.current_xyz, self.current_rgb, self.current_intrinsics, self.current_image.shape)

        # 이전 결과 초기화
        self.instances = []
        self.inst_table.setRowCount(0)
        self.selected_idx = None
        self.target_pose = None
        # view_3d가 곧 clear되니 추적 리스트도 비움 (다음 시각화 시 stale 이름 참조 방지)
        self._instance_actor_names.clear()
        self._grasp_marker_names.clear()
        self._tcp_viz_actors.clear()
        self.btn_move.setEnabled(False)
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.tcp_labels[axis].setText("---")

        self.main.statusBar().showMessage(f"캡처 완료: {image.shape[1]}x{image.shape[0]}")

    def _on_roi_dragged(self, x1: int, y1: int, x2: int, y2: int):
        if self.current_xyz is None:
            return
        self.roi_2d = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.main.statusBar().showMessage(f"ROI 설정: {self.roi_2d}")

    def _clear_roi(self):
        self.roi_2d = None
        self.view_2d.set_roi(None)
        # 3D ROI 박스 제거
        try:
            self.view_3d.plotter.remove_actor("roi_box")
        except Exception:
            pass
        self.view_3d.plotter.render()
        self.main.statusBar().showMessage("ROI 해제")

    def _resolve_roi(self, warn_msg: str):
        """
        처리 대상 ROI 결정. roi_2d가 있으면 그대로, 없으면 확인 후 전체 이미지.
        사용자가 확인 다이얼로그에서 취소하면 None 반환 (호출자는 중단해야 함).
        (_preview_clusters / _run_matching 중복 블록을 하나로)
        """
        if self.roi_2d is not None:
            return self.roi_2d
        ret = QMessageBox.question(
            self, "ROI 미설정", warn_msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return None
        h, w = self.current_xyz.shape[:2]
        return (0, 0, w, h)

    def _preview_clusters(self):
        """
        ROI 안 포인트클라우드를 (옵션) 작업대 평면 제거 후 DBSCAN으로 클러스터링하고
        클러스터 전용 뷰(view_cluster)에 색깔별로 표시.
        매칭 전에 eps/min_pts 파라미터를 시각적으로 조정하기 위한 도구.
        """
        if self.current_xyz is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return

        roi = self._resolve_roi("ROI가 없으면 전체 이미지에 클러스터링이 적용됩니다. 계속?")
        if roi is None:
            return

        scene = crop_pointcloud_by_2d_roi(self.current_xyz, self.current_rgb, roi)
        if scene is None:
            QMessageBox.warning(self, "오류", "ROI 안에 유효한 3D 포인트가 부족합니다")
            return

        # 작업대 평면 제거 (옵션)
        plane_removed_count = 0
        if self.remove_plane_check.isChecked():
            scene_no_plane, _plane_model, plane_removed_count = remove_table_plane(
                scene, distance_threshold=float(self.plane_dist_spin.value())
            )
            scene = scene_no_plane

        eps = float(self.dbscan_eps.value())
        min_pts = int(self.dbscan_min_pts.value())

        self.main.statusBar().showMessage(f"클러스터링 중... scene={len(scene.points)}점")
        QApplication.processEvents()
        clusters = cluster_scene_dbscan(scene, eps=eps, min_points=min_pts)

        # 클러스터 뷰로 전환
        self._switch_view(3)
        plotter = self.view_cluster.plotter
        plotter.clear()

        if not clusters:
            msg = f"DBSCAN 결과: 클러스터 0개 (eps={eps}mm, min_pts={min_pts})"
            if plane_removed_count > 0:
                msg += f"\n작업대 평면 제거: {plane_removed_count}점"
            msg += "\n\n조치: eps를 늘리거나(점들 사이 거리가 큼), min_pts를 줄이세요."
            QMessageBox.information(self, "결과", msg)
            return

        # 각 클러스터를 다른 색으로 시각화 (공용 팔레트 재사용)
        cluster_sizes = []
        for ci, cluster in enumerate(clusters):
            pts = np.asarray(cluster.points, dtype=np.float32)
            if len(pts) == 0:
                continue
            color = INSTANCE_COLORS_RGB[ci % len(INSTANCE_COLORS_RGB)]
            color01 = tuple(c / 255.0 for c in color)
            plotter.add_mesh(
                pv.PolyData(pts),
                color=color01,
                point_size=4,
                render_points_as_spheres=False,
                name=f"cluster_{ci}",
                pickable=False,
                reset_camera=False,
            )
            # 라벨 (클러스터 중심)
            center = pts.mean(axis=0)
            plotter.add_point_labels(
                np.array([center], dtype=np.float32),
                [f"#{ci + 1} ({len(pts)}점)"],
                point_size=1,
                font_size=14,
                text_color="white",
                name=f"cluster_label_{ci}",
                always_visible=True,
                pickable=False,
                show_points=False,
            )
            cluster_sizes.append(len(pts))

        # 카메라 설정
        all_pts = np.vstack([np.asarray(c.points, dtype=np.float32) for c in clusters])
        self.view_cluster.set_camera_from_intrinsics(
            all_pts,
            self.current_intrinsics,
            self.current_image.shape if self.current_image is not None else None,
        )
        plotter.render()

        size_str = ", ".join(str(s) for s in cluster_sizes)
        plane_msg = f", 평면제거={plane_removed_count}점" if plane_removed_count > 0 else ""
        self.main.statusBar().showMessage(
            f"클러스터 {len(clusters)}개: [{size_str}]{plane_msg} (eps={eps}mm, min_pts={min_pts})"
        )

    def _preview_cad(self):
        """
        CAD 전용 3D 뷰(view_cad)에 CAD를 원점에 띄움 (좌표축과 함께).

        Zivid 포인트클라우드 뷰(view_3d)와 별도 위젯이므로 카메라 시점/리소스 충돌 없음.

        시각화:
          - 빨강 X축, 초록 Y축, 파랑 Z축 (50mm 길이)
          - 회색 = 원본 CAD 전체
          - 빨강 점 = 현재 cull 옵션으로 잘라낸 "보이는 면"
          - 노랑 점 = voxel 다운샘플 결과 (실제 매칭에 쓰이는 점들)
        """
        if self.cad_pcd is None:
            QMessageBox.warning(self, "오류", "CAD 모델을 먼저 로드하세요")
            return

        # CAD 전용 뷰로 전환
        self._switch_view(2)

        plotter = self.view_cad.plotter
        plotter.clear()

        # 원본 CAD (회색)
        all_pts = np.asarray(self.cad_pcd.points, dtype=np.float32)
        if len(all_pts) > 0:
            plotter.add_mesh(
                pv.PolyData(all_pts),
                color="lightgray",
                point_size=2,
                render_points_as_spheres=False,
                name="cad_full",
                pickable=False,
                reset_camera=False,
            )

        # cull 적용 여부 (체크박스 + 잡기 축 상태에 따름)
        grasp_axis_text = self.grasp_axis_combo.currentText()
        is_grasp_off = grasp_axis_text.startswith("Off")
        cull_active = self.cull_visible_check.isChecked() and not is_grasp_off

        if cull_active:
            view_axis = "+" + grasp_axis_text
            source_for_match = cull_model_to_visible(self.cad_pcd, view_axis)
            culled_pts = np.asarray(source_for_match.points, dtype=np.float32)
            if len(culled_pts) > 0:
                plotter.add_mesh(
                    pv.PolyData(culled_pts),
                    color="#ff5050",
                    point_size=4,
                    render_points_as_spheres=True,
                    name="cad_culled",
                    pickable=False,
                    reset_camera=False,
                )
            cull_label = f"cull {view_axis}"
            n_culled = len(culled_pts)
        else:
            # cull 안 함 → 매칭에 원본 전체 사용
            source_for_match = self.cad_pcd
            cull_label = "cull 미적용 (원본 전체)" if is_grasp_off else "cull 미적용"
            n_culled = 0

        # 다운샘플 결과 (노랑) - 매칭 시 RANSAC/FPFH가 실제 쓰는 점들. voxel에 영향 받음
        voxel = float(self.voxel_spin.value())
        n_down = 0
        if voxel > 0 and len(source_for_match.points) > 0:
            try:
                source_down = source_for_match.voxel_down_sample(voxel)
                down_pts = np.asarray(source_down.points, dtype=np.float32)
                n_down = len(down_pts)
                if n_down > 0:
                    plotter.add_mesh(
                        pv.PolyData(down_pts),
                        color="#ffd400",
                        point_size=10,
                        render_points_as_spheres=True,
                        name="cad_downsampled",
                        pickable=False,
                        reset_camera=False,
                    )
            except Exception as e:
                logger.warning(f"미리보기 다운샘플 실패: {e}")

        # 좌표축 (50mm 길이)
        L = 50.0
        origin = np.array([0, 0, 0], dtype=np.float32)
        for axis_pts, color, name in [
            (np.array([origin, [L, 0, 0]], dtype=np.float32), "red", "axis_x"),
            (np.array([origin, [0, L, 0]], dtype=np.float32), "green", "axis_y"),
            (np.array([origin, [0, 0, L]], dtype=np.float32), "blue", "axis_z"),
        ]:
            line = pv.PolyData(axis_pts)
            line.lines = np.array([2, 0, 1])
            plotter.add_mesh(
                line,
                color=color,
                line_width=5,
                name=name,
                pickable=False,
                reset_camera=False,
                render_lines_as_tubes=True,
            )

        # 축 라벨
        plotter.add_point_labels(
            np.array([[L + 5, 0, 0], [0, L + 5, 0], [0, 0, L + 5]], dtype=np.float32),
            ["X", "Y", "Z"],
            point_size=1,
            font_size=18,
            text_color="white",
            name="axis_labels",
            always_visible=True,
            pickable=False,
            show_points=False,
        )

        # 카메라 자동 맞춤
        plotter.reset_camera()
        plotter.render()

        # Grasp 위치 마커 (녹색 구 + 잡기 축 방향 화살표)
        self._update_grasp_marker_in_cad_view()

        n_full = len(all_pts)
        cull_part = f"빨강({cull_label})={n_culled}점, " if cull_active else f"({cull_label}, 빨강 표시 X), "
        gx, gy, gz = self.grasp_position_cad
        self.main.statusBar().showMessage(
            f"CAD 미리보기: 원본={n_full}점, {cull_part}노랑(voxel {voxel}mm 다운샘플)={n_down}점, "
            f"Grasp(녹색)=({gx:.1f}, {gy:.1f}, {gz:.1f})."
        )

    # ---------------------------------------------------------
    # 매칭 실행
    # ---------------------------------------------------------

    def _run_matching(self):
        if self.cad_pcd is None:
            QMessageBox.warning(self, "오류", "CAD 모델을 먼저 로드하세요")
            return
        if self.current_xyz is None:
            QMessageBox.warning(self, "오류", "캡처를 먼저 하세요")
            return
        roi = self._resolve_roi(
            "ROI가 설정되지 않았습니다. 전체 이미지에서 매칭을 시도하면 시간이 오래 걸릴 수 있습니다. 계속하시겠습니까?"
        )
        if roi is None:
            return

        # ROI 내 포인트 클라우드 추출
        scene = crop_pointcloud_by_2d_roi(self.current_xyz, self.current_rgb, roi)
        if scene is None or len(scene.points) < 100:
            QMessageBox.warning(self, "오류", "ROI 안에 유효한 3D 포인트가 부족합니다")
            return

        # 작업대 평면 제거 (옵션)
        plane_msg = "작업대 평면 제거 사용 안 함"
        if self.remove_plane_check.isChecked():
            scene_no_plane, _plane_model, plane_n = remove_table_plane(
                scene, distance_threshold=float(self.plane_dist_spin.value())
            )
            scene = scene_no_plane
            plane_msg = f"작업대 평면 제거: {plane_n}점 (남은 scene {len(scene.points)}점)"
            if len(scene.points) < 100:
                QMessageBox.warning(self, "오류", "평면 제거 후 남은 점이 너무 적습니다. 평면 두께를 줄이거나 옵션을 끄세요.")
                return

        voxel = float(self.voxel_spin.value())
        max_inst = int(self.max_inst_spin.value())
        fit_thr = float(self.fitness_spin.value())

        # CAD: 옵션에 따라 보이는 면만 추출 (잡기 축 Off면 cull도 자동 끔)
        grasp_axis_text = self.grasp_axis_combo.currentText()
        is_grasp_off = grasp_axis_text.startswith("Off")
        if self.cull_visible_check.isChecked() and not is_grasp_off:
            view_axis = "+" + grasp_axis_text
            cad_for_match = cull_model_to_visible(self.cad_pcd, view_axis)
            cull_msg = f"CAD cull ({view_axis}): {len(self.cad_pcd.points)} → {len(cad_for_match.points)}점"
            logger.info(cull_msg)
        else:
            cad_for_match = self.cad_pcd
            if is_grasp_off:
                cull_msg = "CAD cull 사용 안 함 (잡기 축 Off)"
            else:
                cull_msg = "CAD cull 사용 안 함"

        # 진행 표시
        self.progress.setVisible(True)
        self.progress.setRange(0, max_inst)
        self.progress.setValue(0)
        self.btn_match.setEnabled(False)
        self.main.statusBar().showMessage(f"매칭 중... scene={len(scene.points)}점, voxel={voxel}mm")
        QApplication.processEvents()

        def progress_cb(i, total, msg):
            self.progress.setValue(i)
            self.main.statusBar().showMessage(msg)
            QApplication.processEvents()

        try:
            algo = self.algo_combo.currentText()
            use_fgr = "(FGR)" in algo
            if algo.startswith("PPF"):
                # PPF detector 학습 (첫 호출에만 실제 학습, 이후 캐시)
                if self._ppf_detector is None or self._ppf_cad_path != self.cad_path:
                    self.main.statusBar().showMessage("PPF 학습 중... (수~수십초 소요)")
                    QApplication.processEvents()
                    detector, model_data, model_o3d = train_ppf_detector(self.cad_pcd)
                    self._ppf_detector = detector
                    self._ppf_model_data = model_data
                    self._ppf_model_o3d = model_o3d
                    self._ppf_cad_path = self.cad_path
                    logger.info(f"PPF 학습 완료: model {len(model_data)}점")

                # PPF는 DBSCAN 클러스터링 사용 (꺼져 있으면 단일 클러스터로 처리되도록 강제)
                eps = float(self.dbscan_eps.value()) if self.use_dbscan.isChecked() else 1e6
                min_pts = int(self.dbscan_min_pts.value()) if self.use_dbscan.isChecked() else 10

                instances, debug_log = ppf_match_per_cluster(
                    scene,
                    self._ppf_detector,
                    self._ppf_model_data,
                    self._ppf_model_o3d,
                    eps=eps,
                    min_points=min_pts,
                    progress_cb=progress_cb,
                )
            elif self.use_dbscan.isChecked():
                instances, debug_log = cad_match_per_cluster(
                    scene,
                    cad_for_match,
                    voxel,
                    eps=float(self.dbscan_eps.value()),
                    min_points=int(self.dbscan_min_pts.value()),
                    fitness_threshold=fit_thr,
                    ransac_attempts=int(self.ransac_attempts_spin.value()),
                    use_fgr=use_fgr,
                    progress_cb=progress_cb,
                )
            else:
                instances, debug_log = cad_match_multi_instance(
                    scene,
                    cad_for_match,
                    voxel,
                    max_instances=max_inst,
                    fitness_threshold=fit_thr,
                    ransac_attempts=int(self.ransac_attempts_spin.value()),
                    use_fgr=use_fgr,
                    progress_cb=progress_cb,
                )
        except Exception as e:
            self.progress.setVisible(False)
            self.btn_match.setEnabled(True)
            QMessageBox.critical(self, "매칭 오류", f"매칭 중 오류:\n{e}")
            logger.exception("매칭 실패")
            return

        self.progress.setVisible(False)
        self.btn_match.setEnabled(True)

        self.instances = instances
        logger.info(f"매칭 완료: {len(instances)}개 인스턴스 발견")

        # UI 업데이트
        self._update_instance_table()
        self._render_instances_3d()

        # 진단 로그를 다이얼로그에 표시
        log_text = plane_msg + "\n" + cull_msg + "\n" + "\n".join(debug_log)
        if not instances:
            full_msg = (
                "매칭된 인스턴스가 없습니다.\n\n"
                "=== 진단 로그 ===\n"
                f"{log_text}\n\n"
                "=== 조치 가이드 ===\n"
                "• RANSAC fitness=0 이면 → CAD/scene 형상 매칭 자체가 안됨. "
                "voxel size를 객체 디테일 크기로 조정 (CAD 대각선/30~60).\n"
                "• ICP fitness가 임계값보다 살짝 낮으면 → 임계값 spin을 더 낮춰보기 (0.10~0.15).\n"
                "• ICP fitness=0 이면 → RANSAC 자세가 완전히 빗나감. ROI를 더 좁히거나 voxel을 바꿔보기.\n"
                "• '보이는 면만 사용' 체크박스가 켜져 있는지 확인.\n"
                "• '잡기 축'이 CAD에서 위쪽 방향과 일치하는지 확인 (CAD를 별도 뷰어로 열어 확인)."
            )
            QMessageBox.information(self, "결과 (매칭 실패)", full_msg)
        else:
            full_msg = f"매칭 성공: {len(instances)}개 인스턴스\n\n=== 진단 로그 ===\n{log_text}"
            QMessageBox.information(self, f"결과: {len(instances)}개 매칭", full_msg)
            self.main.statusBar().showMessage(f"매칭 완료: {len(instances)}개 발견 (voxel={voxel}mm)")

    def _update_instance_table(self):
        self.inst_table.setRowCount(0)
        for i, inst in enumerate(self.instances):
            T = inst["transformation"]
            pos = T[:3, 3]
            row = self.inst_table.rowCount()
            self.inst_table.insertRow(row)
            self.inst_table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self.inst_table.setItem(row, 1, QTableWidgetItem(f"{inst['fitness']:.3f}"))
            self.inst_table.setItem(row, 2, QTableWidgetItem(f"{inst['rmse']:.2f}"))
            self.inst_table.setItem(row, 3, QTableWidgetItem(f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})"))

    def _render_instances_3d(self):
        """3D 뷰에 매칭된 인스턴스를 다른 색 메시(또는 포인트)로 표시"""
        plotter = self.view_3d.plotter

        # 직전에 추가한 actor만 정확히 제거 (range(20) 무차별 호출 제거)
        for name in self._instance_actor_names:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._instance_actor_names.clear()
        # 이전 TCP 시각화도 함께 제거 (새 매칭 결과에서 stale로 남는 것 방지)
        for name in self._tcp_viz_actors:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._tcp_viz_actors.clear()

        if not self.instances:
            self._update_grasp_markers_in_3d_view()
            plotter.render()
            return

        # CAD 원본 정점/면을 한 번만 추출 (인스턴스마다 deepcopy하던 것 제거).
        # 변환은 numpy 행렬곱으로: verts' = verts @ R^T + t
        use_mesh = self.cad_mesh is not None
        if use_mesh:
            base_verts = np.asarray(self.cad_mesh.vertices, dtype=np.float64)
            tris = np.asarray(self.cad_mesh.triangles)
            faces_pv = (
                np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).flatten()
                if len(tris) > 0
                else None
            )
        else:
            base_verts = np.asarray(self.cad_pcd.points, dtype=np.float64)
            faces_pv = None

        for i, inst in enumerate(self.instances):
            T = inst["transformation"]
            color01 = tuple(c / 255.0 for c in INSTANCE_COLORS_RGB[i % len(INSTANCE_COLORS_RGB)])

            if len(base_verts) > 0:
                verts_t = (base_verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
                if use_mesh and faces_pv is not None:
                    pv_obj = pv.PolyData(verts_t, faces_pv)
                    plotter.add_mesh(
                        pv_obj, color=color01, opacity=0.55,
                        name=f"inst_{i}", pickable=False, reset_camera=False,
                    )
                else:
                    pv_obj = pv.PolyData(verts_t)
                    plotter.add_mesh(
                        pv_obj, color=color01, point_size=4, render_points_as_spheres=True,
                        name=f"inst_{i}", pickable=False, reset_camera=False,
                    )
                self._instance_actor_names.append(f"inst_{i}")

            # 객체 좌표축 (40mm)
            origin = T[:3, 3]
            axes_pts = np.array([origin, origin + T[:3, 0] * 40, origin + T[:3, 1] * 40, origin + T[:3, 2] * 40])
            axes_poly = pv.PolyData(axes_pts.astype(np.float32))
            axes_poly.lines = np.array([[2, 0, 1], [2, 0, 2], [2, 0, 3]]).flatten()
            plotter.add_mesh(
                axes_poly, color=color01, line_width=4,
                name=f"inst_axis_{i}", pickable=False, reset_camera=False, render_lines_as_tubes=True,
            )
            self._instance_actor_names.append(f"inst_axis_{i}")

            # 라벨
            plotter.add_point_labels(
                np.array([origin + np.array([0, 0, -15], dtype=np.float32)]),
                [f"#{i + 1}"],
                point_size=1, font_size=14, text_color="white",
                name=f"inst_label_{i}", always_visible=True, pickable=False, show_points=False,
            )
            self._instance_actor_names.append(f"inst_label_{i}")

        self._update_grasp_markers_in_3d_view()
        plotter.render()

    # ---------------------------------------------------------
    # 객체 선택 → TCP 자세
    # ---------------------------------------------------------

    def _on_table_selection(self):
        rows = self.inst_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        self._select_instance(idx)

    def _select_instance(self, idx: int, silent: bool = False):
        """
        인스턴스 선택 → TCP 자세 계산.
        silent=True: 스핀박스 valueChanged 등에서 호출 시 모달 경고를 띄우지 않음
                     (매 틱마다 팝업이 쌓이는 것을 방지).
        """
        if idx < 0 or idx >= len(self.instances):
            return
        if self.T_calib is None:
            if not silent:
                QMessageBox.warning(self, "오류", "캘리브레이션을 먼저 로드하세요")
            return

        # 다른 인스턴스로 새로 선택하면 180° flip 상태 초기화 (객체마다 독립).
        # silent 재계산(grasp 스핀 변경 등)은 같은 인스턴스이므로 flip 유지.
        if idx != self.selected_idx:
            self._flip_applied = False

        self.selected_idx = idx
        inst = self.instances[idx]
        T_obj_cam = inst["transformation"]

        cur_tcp = None
        if self.main.robot:
            cur_tcp = self.main.robot.get_tcp_position()

        tcp = object_pose_to_tcp(
            T_obj_cam,
            self.T_calib,
            self.calib_mode,
            cur_tcp,
            grasp_axis=self.grasp_axis_combo.currentText(),
            grasp_flip=self.grasp_flip_check.isChecked(),
            grasp_offset_xyz=tuple(self.grasp_position_cad),
            grasp_rotation_abc_deg=tuple(self.grasp_rotation_abc_deg),
        )

        if tcp is None:
            for axis in ["X", "Y", "Z", "A", "B", "C"]:
                self.tcp_labels[axis].setText("---")
            self.btn_move.setEnabled(False)
            return

        # 180° flip이 적용 상태면 Tool +Z축 둘레 회전을 여기서 적용 (영속).
        # 이렇게 하면 grasp 스핀 변경 등으로 재계산돼도 flip이 유지됨.
        if self._flip_applied:
            T = tcp_to_homogeneous(tcp)
            R_flip = np.eye(4)
            R_flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
            tcp = homogeneous_to_tcp(T @ R_flip)

        self.target_pose = tcp
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            self.tcp_labels[axis].setText(f"{tcp[axis.lower()]:.2f}")

        connected = self.main.robot is not None
        self.btn_move.setEnabled(connected)
        self.btn_add_obj_to_seq.setEnabled(connected)
        flip_str = " [180° 적용]" if self._flip_applied else ""
        self.main.statusBar().showMessage(
            f"인스턴스 #{idx + 1} 선택{flip_str}: TCP X={tcp['x']:.1f}, Y={tcp['y']:.1f}, Z={tcp['z']:.1f}"
        )

        # 3D 뷰에 Tool 자세 시각화 (Tool 좌표축 + approach 지점 + 경로선)
        self._render_tcp_visualization()

    def _render_tcp_visualization(self):
        """
        선택된 인스턴스의 grasp 점에 그리퍼 접근 자세를 시각화:
          - Tool 좌표축 (X 빨강, Y 초록, Z 파랑) → 그리퍼가 어느 방향으로 다가갈지
          - Approach 지점 (주황 구) + target까지 경로선

        bin_picking 탭의 동일 기능과 같은 디자인. 단 cad_matching은
        target_pose의 origin이 객체 중심이 아니라 grasp 점이고, ABC 회전
        보정도 들어가 있으므로:
          - 위치는 인스턴스 변환 + grasp_offset을 카메라 좌표계로 적용한 점
          - 회전은 target_pose(베이스)를 카메라 좌표계로 역변환
        """
        plotter = self.view_3d.plotter
        for name in self._tcp_viz_actors:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self._tcp_viz_actors.clear()

        if (
            self.target_pose is None
            or self.selected_idx is None
            or self.T_calib is None
            or self.selected_idx >= len(self.instances)
        ):
            plotter.render()
            return

        # 위치: 인스턴스 변환 + grasp_offset → 카메라 좌표계 grasp 점
        T_inst = self.instances[self.selected_idx]["transformation"]
        grasp_local = np.array([
            self.grasp_position_cad[0],
            self.grasp_position_cad[1],
            self.grasp_position_cad[2],
            1.0,
        ])
        origin = (T_inst @ grasp_local)[:3].astype(np.float32)

        # 회전: 베이스 좌표계 target 자세 → 카메라 좌표계
        R_target_base = tcp_to_homogeneous(self.target_pose)[:3, :3]
        if self.calib_mode == "eye_to_hand":
            R_in_cam = self.T_calib[:3, :3].T @ R_target_base
        elif self.calib_mode == "eye_in_hand":
            if self.main.robot is None:
                plotter.render()
                return
            cur_tcp = self.main.robot.get_tcp_position()
            T_g2b = tcp_to_homogeneous(cur_tcp)
            R_in_cam = self.T_calib[:3, :3].T @ T_g2b[:3, :3].T @ R_target_base
        else:
            plotter.render()
            return

        L = 50.0
        for axis_idx, color, suffix in [(0, "red", "x"), (1, "green", "y"), (2, "blue", "z")]:
            endpoint = (origin + R_in_cam[:, axis_idx] * L).astype(np.float32)
            line = pv.PolyData(np.array([origin, endpoint], dtype=np.float32))
            line.lines = np.array([2, 0, 1])
            name = f"tcp_axis_{suffix}"
            plotter.add_mesh(
                line, color=color, line_width=6, name=name,
                render_lines_as_tubes=True, pickable=False, reset_camera=False,
            )
            self._tcp_viz_actors.append(name)

        # Approach 지점 (Tool -Z 방향으로 offset)
        offset = float(self.approach_dist.value()) if self.use_approach.isChecked() else 50.0
        approach_pos = (origin - R_in_cam[:, 2] * offset).astype(np.float32)
        sphere = pv.Sphere(radius=4, center=approach_pos)
        plotter.add_mesh(
            sphere, color="#ffaa00", name="tcp_approach", pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("tcp_approach")

        # Approach → Target 경로선
        path = pv.PolyData(np.array([approach_pos, origin], dtype=np.float32))
        path.lines = np.array([2, 0, 1])
        plotter.add_mesh(
            path, color="#ffaa00", line_width=3, name="tcp_path",
            render_lines_as_tubes=True, pickable=False, reset_camera=False,
        )
        self._tcp_viz_actors.append("tcp_path")

        plotter.render()

    def _flip_target_180(self):
        """
        Tool +Z축 둘레 180° 회전을 토글 (영속 상태).
        잡는 위치는 그대로, 그리퍼 방향만 반대 (좌우 대칭 객체 매칭 모호성 보정).
        _flip_applied 플래그를 뒤집고 _select_instance가 실제 적용 → grasp 스핀
        변경/재계산에도 flip이 사라지지 않음.
        """
        if self.selected_idx is None or self.target_pose is None:
            QMessageBox.warning(self, "오류", "먼저 인스턴스를 선택하세요")
            return
        self._flip_applied = not self._flip_applied
        self._select_instance(self.selected_idx, silent=True)
        state = "적용됨" if self._flip_applied else "해제됨"
        self.main.statusBar().showMessage(f"Tool +Z축 둘레 180° 회전 {state} (다시 누르면 토글)")

    # ---------------------------------------------------------
    # 로봇 이동
    # ---------------------------------------------------------

    def _execute_move(self):
        if self.target_pose is None:
            QMessageBox.warning(self, "오류", "객체를 먼저 선택하세요")
            return
        if self.main.robot is None:
            QMessageBox.warning(self, "오류", "로봇이 연결되지 않았습니다")
            return

        p = self.target_pose
        if not self._validate_z(p["z"]):
            return

        is_lin = self.move_mode_combo.currentText().startswith("LIN")
        speed = self._effective_speed(self.speed_spin.value())
        use_approach = self.use_approach.isChecked()
        offset = self.approach_dist.value()

        # Approach 위치 계산 (시퀀스 경로와 동일한 헬퍼 사용 → 분기 위험 제거)
        approach_xyz = None
        if use_approach:
            approach_xyz = self._compute_approach_position(p, offset)
            if not self._validate_z(approach_xyz[2]):
                return

        # 확인 다이얼로그
        if use_approach:
            msg = (
                f"⚠ CAD 매칭 인스턴스 #{self.selected_idx + 1} 이동\n\n"
                f"방식: {'LIN' if is_lin else 'PTP'}, 속도: {speed}%\n"
                f"Approach offset: {offset}mm\n\n"
                f"[1] Approach: ({approach_xyz[0]:.1f}, {approach_xyz[1]:.1f}, {approach_xyz[2]:.1f})\n"
                f"[2] Target  : ({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f}) "
                f"A={p['a']:.1f}, B={p['b']:.1f}, C={p['c']:.1f}\n"
                f"[3] Retract : (Approach 동일)\n\n"
                f"진행하시겠습니까?"
            )
        else:
            msg = (
                f"⚠ CAD 매칭 인스턴스 #{self.selected_idx + 1} 이동\n\n"
                f"방식: {'LIN' if is_lin else 'PTP'}, 속도: {speed}%\n"
                f"목표: ({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f}) "
                f"A={p['a']:.1f}, B={p['b']:.1f}, C={p['c']:.1f}\n\n"
                f"진행하시겠습니까?"
            )

        ret = QMessageBox.question(self, "이동 확인", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        try:
            self.main.robot.set_speed(speed)

            def add_motion(x, y, z, a, b, c):
                if is_lin:
                    return self.main.robot.add_move_lin(x, y, z, a, b, c)
                return self.main.robot.add_move_ptp(x, y, z, a, b, c)

            slots = []
            if use_approach:
                ax, ay, az = approach_xyz
                s1 = add_motion(ax, ay, az, p["a"], p["b"], p["c"])
                if s1 is None:
                    QMessageBox.critical(self, "오류", "Approach 모션 큐 추가 실패")
                    return
                slots.append(s1)
                s2 = self.main.robot.add_move_lin(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])
                if s2 is None:
                    QMessageBox.critical(self, "오류", "Target 모션 큐 추가 실패")
                    return
                slots.append(s2)
                s3 = self.main.robot.add_move_lin(ax, ay, az, p["a"], p["b"], p["c"])
                if s3 is None:
                    QMessageBox.critical(self, "오류", "Retract 모션 큐 추가 실패")
                    return
                slots.append(s3)
                self.main.statusBar().showMessage(f"3단계 모션 큐 추가: slots={slots}")
            else:
                s = add_motion(p["x"], p["y"], p["z"], p["a"], p["b"], p["c"])
                if s is None:
                    QMessageBox.critical(self, "오류", "큐 추가 실패")
                    return
                slots.append(s)
                self.main.statusBar().showMessage(f"단일 모션 큐 추가: slot={s}")
            logger.info(f"CAD 매칭 이동: inst={self.selected_idx}, slots={slots}, target={p}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이동 명령 실패:\n{e}")
            logger.exception("이동 실패")

    # ---------------------------------------------------------
    # 모드 폴링
    # ---------------------------------------------------------

    def _refresh_mode_display(self):
        if not self.main.robot:
            self.mode_label.setText("모드: 미연결")
            self.mode_label.setStyleSheet("padding: 4px 10px; font-weight: bold; background-color: #BDBDBD; color: white; border-radius: 3px;")
            self._current_mode = "?"
            return
        try:
            raw = self.main.robot.read_variable("$MODE_OP")
            if raw is None:
                return
            mode = normalize_robot_mode(raw)
            self._current_mode = mode
            self.mode_label.setText(f"모드: {mode}")
            if is_auto_mode(mode):
                bg = "#d32f2f"
            elif mode == "T1":
                bg = "#388e3c"
            elif mode == "T2":
                bg = "#f57c00"
            else:
                bg = "#616161"
            self.mode_label.setStyleSheet(f"padding: 4px 10px; font-weight: bold; background-color: {bg}; color: white; border-radius: 3px;")
        except Exception:
            pass
