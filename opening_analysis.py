"""
여는 방향 / OBB 분석 (순수 CV, Qt 독립)
=======================================
클램셸 화장품 케이스의 여는 방향을 추정하는 알고리즘 모음. bin_picking_tab 에서
분리해 앱 없이도 단독 테스트/튜닝이 가능하게 했다. UI(버튼·상태바·다이얼로그)는
탭에 남고, 이 모듈은 마스크·이미지·OBB 만 받아 결과 dict 를 돌려준다.

주요 함수:
  - obb_from_mask(mask) → {center, size, angle, box_pts}
  - opening_weight_map(gray, method, thr_pct) → 가중치 맵 (seam/brightness)
  - opening_from_weight(mask, weight, obb, ...) → {dir, angle_deg, confidence, ...}
  - opening_from_grid(mask, gray, obb, ...)      → 내부 격자 비대칭 (권장, 투명체)
  - debug_show_opening / debug_show_grid         → 개발용 cv2.imshow 시각화

자세한 원리는 docs/bin_picking.md §3.3/§3.4 참고.
"""

import logging
from typing import Optional, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def obb_from_mask(mask) -> Optional[Dict]:
    """마스크(H×W)에서 cv2.minAreaRect 로 OBB 딕셔너리 계산. 실패 시 None."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    (cx, cy), (rw, rh), angle = rect
    box_pts = cv2.boxPoints(rect)  # (4,2) float
    return {
        "center": (float(cx), float(cy)),
        "size": (float(rw), float(rh)),
        "angle": float(angle),
        "box_pts": box_pts.tolist(),
    }


def opening_from_weight(mask, weight, obb, erode_ratio: float = 0.06, debug: bool = False) -> Optional[Dict]:
    """가중치 맵의 무게중심 비대칭으로 클램셸의 여는 방향을 추정한다 (방식 공통).

    원리: 힌지와 립(여는 쪽)은 둘 다 **긴 변**이므로, 여는 방향은 두 긴 변
    사이 = **짧은 extent 축(단축)** 을 향한다. 여는 축은 단축으로 고정하고,
    가중치 무게중심의 오프셋 **부호(둘 중 어느 긴 변이 립인가)** 만 정한다.
    weight 는 방식마다 다르다: 이음선=에지 크기, 내부=밝기. 외곽 실루엣의
    영향은 마스크를 침식(erode)해 제외 → 내부만 반영.

    반환: {"dir"(단위벡터), "angle_deg", "confidence"(단축 오프셋/반길이 비),
           "half_len"(단축 반길이), "axis": "short"} — 실패 시 None.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    pts = np.asarray(obb["box_pts"], dtype=np.float64)  # (4,2) 순서대로 인접
    e1 = pts[1] - pts[0]
    e2 = pts[2] - pts[1]
    l1 = float(np.linalg.norm(e1))
    l2 = float(np.linalg.norm(e2))
    if l1 < 2 or l2 < 2:
        return None
    u1 = e1 / l1
    u2 = e2 / l2  # 서로 수직인 두 OBB 축 단위벡터

    # 외곽 실루엣 제외: 마스크를 짧은 변의 erode_ratio 만큼 침식
    k = int(round(min(l1, l2) * max(0.0, erode_ratio)))
    eroded = mask_u8
    if k >= 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        e = cv2.erode(mask_u8, ker)
        if int(e.sum()) >= 20:
            eroded = e  # 너무 작아지면 침식 생략
    ys, xs = np.nonzero(eroded)
    if len(xs) < 20:
        return None
    w = weight[ys, xs].astype(np.float64)
    w = np.clip(w, 0.0, None)  # 음수 가중치 방지
    wsum = float(w.sum())
    if wsum < 1e-6:
        return None

    gx0 = float(xs.mean())
    gy0 = float(ys.mean())  # 지지영역 기하 중심 (오프셋 0 기준)
    ex = float((xs * w).sum() / wsum)
    ey = float((ys * w).sum() / wsum)  # 가중 무게중심
    off = np.array([ex - gx0, ey - gy0])

    # 여는 축 = 짧은 extent 축으로 고정. u1 은 길이 l1 변을 따라가므로 그 축의
    # extent 는 l1 → extent 가 작은 쪽이 단축.
    if l1 <= l2:
        axis, proj, half = u1, float(off.dot(u1)), l1 / 2.0
    else:
        axis, proj, half = u2, float(off.dot(u2)), l2 / 2.0
    ratio = abs(proj) / half  # 단축 반길이로 정규화한 오프셋
    s = 1.0 if proj >= 0 else -1.0
    d = axis * s  # 여는 방향(무게중심이 치우친 쪽)
    result = {
        "dir": (float(d[0]), float(d[1])),
        "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
        "confidence": float(ratio),
        "half_len": float(half),
        "axis": "short",
    }
    if debug:  # 개발용 디버그 시각화가 쓸 중간 산출물
        result["_debug"] = {"eroded": eroded, "geom": (gx0, gy0), "wc": (ex, ey)}
    return result

# ---- 개발용 디버그 시각화 (OPENING_DEBUG=True 일 때만) ----

def _label_hconcat(panels, h: int = 260, pad: int = 6):
    """[(라벨, BGR 이미지), ...] → 같은 높이로 리사이즈 + 라벨바 붙여 가로로 이어붙인 몽타주."""
    cols = []
    for name, p in panels:
        if p is None or getattr(p, "size", 0) == 0:
            continue
        ph, pw = p.shape[:2]
        r = cv2.resize(p, (max(1, int(pw * (h / ph))), h))
        bar = np.full((24, r.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, name, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cols.append(np.vstack([bar, r]))
    if not cols:
        return None
    hmax = max(c.shape[0] for c in cols)
    out = []
    for c in cols:
        if c.shape[0] < hmax:
            c = np.vstack([c, np.full((hmax - c.shape[0], c.shape[1], 3), 40, np.uint8)])
        out.append(c)
        out.append(np.zeros((hmax, pad, 3), np.uint8))
    return np.hstack(out[:-1])

def debug_show_opening(rgb, det, obb, weight, res, idx):
    """'여는 방향' 영상처리 단계를 cv2.imshow 몽타주로 표시 (개발용, OPENING_DEBUG)."""
    try:
        if rgb is None:
            return
        dbg = res.get("_debug")
        if dbg is None:
            return
        rgb = rgb
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        mask = (np.asarray(det["mask"]) > 0).astype(np.uint8)
        eroded = dbg["eroded"]
        H, W = mask.shape

        pts = np.asarray(obb["box_pts"], float)
        x0 = max(0, int(pts[:, 0].min()) - 20)
        x1 = min(W, int(pts[:, 0].max()) + 20)
        y0 = max(0, int(pts[:, 1].min()) - 20)
        y1 = min(H, int(pts[:, 1].max()) + 20)
        crop = lambda a: a[y0:y1, x0:x1]

        # 1) 원본 + OBB(주황)/mask(노랑) 윤곽 + 기하중심(파랑)/가중중심(빨강) + 여는방향(초록)
        over = img.copy()
        cv2.polylines(over, [pts.astype(np.int32).reshape(-1, 1, 2)], True, (0, 140, 255), 2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 255, 255), 1)
        gx0, gy0 = dbg["geom"]
        ex, ey = dbg["wc"]
        cv2.circle(over, (int(gx0), int(gy0)), 4, (255, 0, 0), -1)
        cv2.circle(over, (int(ex), int(ey)), 4, (0, 0, 255), -1)
        cx, cy = obb["center"]
        dx, dy = res["dir"]
        L = res["half_len"]
        cv2.arrowedLine(over, (int(cx), int(cy)), (int(cx + dx * L), int(cy + dy * L)), (0, 255, 0), 2, tipLength=0.25)
        p_orig = crop(over)

        p_mask = cv2.cvtColor(crop(mask * 255), cv2.COLOR_GRAY2BGR)
        p_erode = cv2.cvtColor(crop(eroded * 255), cv2.COLOR_GRAY2BGR)

        w = crop(weight.astype(np.float32))
        wn = cv2.normalize(w, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p_weight = cv2.applyColorMap(wn, cv2.COLORMAP_JET)
        ws = w * crop(eroded).astype(np.float32)
        wsn = cv2.normalize(ws, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p_ws = cv2.applyColorMap(wsn, cv2.COLORMAP_JET)

        panels = [
            (f"#{idx+1} 1.OBB crop", p_orig),
            ("2.mask", p_mask),
            ("3.erode", p_erode),
            ("4.weight(Sobel/bright)", p_weight),
            ("5.weight x erode", p_ws),
        ]
        montage = _label_hconcat(panels)
        if montage is None:
            return
        win = "opening debug (press any key -> next)"
        cv2.imshow(win, montage)
        cv2.waitKey(0)
        cv2.destroyWindow(win)
    except Exception as e:
        logger.warning(f"여는 방향 디버그 시각화 실패: {e}")

def opening_weight_map(gray: np.ndarray, method: str, thr_pct: int) -> np.ndarray:
    """방식별 가중치 맵. seam=Sobel 에지 크기(선택적 백분위 임계), brightness=밝기."""
    if method == "brightness":
        # 내부 밝기 비대칭: 밝은 쪽으로 무게중심이 치우침. 최솟값을 빼 민감도↑.
        g = gray.astype(np.float32)
        return g - float(g.min())
    # seam(기본): 에지 크기
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    if thr_pct > 0:
        # 지지영역이 아니라 전체 이미지 기준 백분위 — 객체별 지지영역은
        # _opening_from_weight 에서 잘리므로 여기선 전역 임계로 약한 에지만 제거
        t = float(np.percentile(grad, thr_pct))
        grad = np.where(grad >= t, grad, 0.0).astype(np.float32)
    return grad

def opening_from_grid(mask, gray, obb, debug: bool = False,
                      band_thr: float = 0.4, side_crop: float = 0.15) -> Optional[Dict]:
    """내부 격자(칸 배열) 비대칭으로 여는 방향 추정 — 투명 케이스 전용, 가장 강건.

    투명 케이스 내부의 2×N 칸 배열이 한쪽으로 치우쳐(빈 여백 띠가 반대쪽에) 있는 걸
    이용한다. 절차:
      1) OBB 로 케이스를 똑바로 세운다(warp, 장축=가로).
      2) 좌우(케이스 옆벽)를 크게, 상하(여백)를 최소로 잘라낸다 — 옆벽의 세로 에지가
         모든 행을 오염시키므로 제거하는 게 핵심.
      3) 세로벽 밀도(|gx|) 행별 프로파일 → 임계 이상 최장 연속구간 = '격자 밴드'.
      4) 격자 밴드의 위/아래(단축 양끝) 여백 중 넓은 쪽으로 방향을 잡는다.
    여는 축은 단축 고정. 반환 형식은 opening_from_weight 와 동일(부호는 반전 토글로 보정).

    조정 파라미터(케이스가 바뀌면 튜닝):
      band_thr  : 격자 밴드 임계(프로파일 최댓값 대비 비, 0~1). 높이면 격자를 좁게 잡음.
      side_crop : 좌우(옆벽) 크롭 비율(0~0.4). 프레임 두께에 맞춰. 상하 크롭은 4% 고정.
    """
    box = np.asarray(obb["box_pts"], np.float32)
    e_ab = box[1] - box[0]
    e_bc = box[2] - box[1]
    l_ab, l_bc = float(np.linalg.norm(e_ab)), float(np.linalg.norm(e_bc))
    if l_ab >= l_bc:  # 장축이 가로가 되도록 src 순서 결정
        src, Wl, Ws = box[[0, 1, 2, 3]], l_ab, l_bc
    else:
        src, Wl, Ws = box[[1, 2, 3, 0]], l_bc, l_ab
    Wl_i, Ws_i = int(round(Wl)), int(round(Ws))
    if Wl_i < 20 or Ws_i < 12:
        return None
    dst = np.array([[0, 0], [Wl_i - 1, 0], [Wl_i - 1, Ws_i - 1], [0, Ws_i - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(gray, M, (Wl_i, Ws_i))

    bx = int(Wl_i * max(0.0, side_crop))  # 좌우 옆벽 제거 (조정 가능)
    by = max(1, int(Ws_i * 0.04))  # 상하 최소 (여백 살림, 고정)
    gi = warp[by : Ws_i - by, bx : Wl_i - bx].astype(np.float32)
    Hi, Wi = gi.shape
    if Hi < 8 or Wi < 8:
        return None
    gx = np.abs(cv2.Sobel(gi, cv2.CV_32F, 1, 0, ksize=3))
    prof = cv2.GaussianBlur(gx.sum(axis=1).reshape(-1, 1), (1, 9), 0).ravel()
    mx = float(prof.max())
    if mx < 1e-6:
        return None
    prof /= mx

    # 격자 밴드 = 임계 이상 최장 연속구간
    thr = band_thr
    above = prof >= thr
    best = (0, -1)
    i = 0
    while i < Hi:
        if above[i]:
            j = i
            while j < Hi and above[j]:
                j += 1
            if (j - 1 - i) > (best[1] - best[0]):
                best = (i, j - 1)
            i = j
        else:
            i += 1
    lo, hi = best
    if hi < lo:
        return None
    v_band = by + (lo + hi) / 2.0  # 전체 warp 좌표
    v_center = Ws_i / 2.0
    sign_v = 1.0 if v_band < v_center else -1.0  # 밴드 반대쪽(넓은 여백)으로
    conf = abs(v_band - v_center) / (Ws_i / 2.0)

    # canonical 두 점을 원본으로 역매핑 → 원본 이미지 단축 방향 벡터
    Minv = cv2.getPerspectiveTransform(dst, src)
    p = np.array([[[Wl_i / 2.0, v_center]], [[Wl_i / 2.0, v_center + sign_v * Ws_i * 0.25]]], np.float32)
    o = cv2.perspectiveTransform(p, Minv)
    d = o[1, 0] - o[0, 0]
    nd = float(np.linalg.norm(d))
    if nd < 1e-6:
        return None
    d = d / nd
    result = {
        "dir": (float(d[0]), float(d[1])),
        "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
        "confidence": float(conf),
        "half_len": float(min(l_ab, l_bc) / 2.0),
        "axis": "short",
    }
    if debug:
        debug_show_grid(warp, bx, by, lo, hi, prof, thr, conf)
    return result

def debug_show_grid(warp, bx, by, lo, hi, prof, thr, conf):
    """[개발용] 격자 방식 중간 단계(warp+격자밴드+프로파일)를 cv2.imshow 로 표시."""
    try:
        Ws_i, Wl_i = warp.shape[:2]
        vis = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR)
        # 격자 밴드(초록), 크롭 영역(노랑)
        cv2.rectangle(vis, (bx, by), (Wl_i - bx, Ws_i - by), (0, 200, 200), 1)
        cv2.rectangle(vis, (bx, by + lo), (Wl_i - bx, by + hi), (0, 255, 0), 1)
        Hi = len(prof)
        gg = np.full((Hi, 100, 3), 30, np.uint8)
        for y in range(Hi):
            cv2.line(gg, (0, y), (int(prof[y] * 95), y), (0, 200, 255), 1)
        cv2.line(gg, (0, lo), (99, lo), (0, 255, 0), 1)
        cv2.line(gg, (0, hi), (99, hi), (0, 255, 0), 1)
        cv2.line(gg, (int(thr * 95), 0), (int(thr * 95), Hi), (80, 80, 255), 1)
        h = max(vis.shape[0], gg.shape[0])

        def padh(a):
            return np.vstack([a, np.full((h - a.shape[0], a.shape[1], 3), 30, np.uint8)]) if a.shape[0] < h else a

        montage = cv2.hconcat([padh(vis), np.zeros((h, 6, 3), np.uint8), padh(gg)])
        montage = cv2.resize(montage, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.putText(montage, f"grid conf={conf:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        win = "grid debug (press any key -> next)"
        cv2.imshow(win, montage)
        cv2.waitKey(0)
        cv2.destroyWindow(win)
    except Exception as e:
        logger.warning(f"격자 디버그 시각화 실패: {e}")
