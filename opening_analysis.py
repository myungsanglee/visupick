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
  - opening_from_grid(mask, gray, obb, ...)      → 내부 격자 비대칭 (가로로 긴 투명 케이스)
  - opening_from_hinge(mask, gray, obb, ...)     → 힌지 투명도 4방향 (정사각형 케이스)
  - debug_show_opening / debug_show_grid / debug_show_hinge → 개발용 cv2.imshow 시각화

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


def opening_from_grid(mask, gray, obb, debug: bool = False, band_thr: float = 0.4, side_crop: float = 0.15) -> Optional[Dict]:
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


# ============================================================
# 힌지 투명도 방식 (정사각형 케이스 — 4방향 판별)
# ============================================================
#
# 왜 별도 방식인가: 위의 세 방식(seam/brightness/grid)은 **여는 축을 OBB 단축으로
# 고정**하고 부호(둘 중 어느 긴 변인가)만 정한다. 가로로 긴 케이스는 힌지와 립이
# 둘 다 긴 변이므로 이게 맞지만, **정사각형 케이스는 네 변 길이가 같아** 단축이
# 노이즈로 정해진다 — 프레임마다 축이 뒤집히고, 애초에 경우의 수가 4가지라
# 2택 구조로는 원리적으로 풀 수 없다.
#
# 그래서 이 방식은 **네 변을 모두 평가**해 힌지를 고르고, 그 반대쪽을 여는 방향으로
# 삼는다. 판별 단서는 "힌지가 있는 변만 투명하다"는 제품 특징이다.
#
# "투명하다"를 이미지에서 어떻게 재나 — **본체 색 대비**:
#   띠의 색이 **케이스 본체 색에서 얼마나 벗어났나**를 Lab 색상면(a*b*) **거리**로 잰다.
#   힌지 쪽은 투명하거나(바닥이 비침) 금속이라 **중성(회색)** 에 가깝고, 나머지 세 변은
#   본체 색(예: 분홍) 그대로다. 점수가 높을수록 힌지.
#
#   밝기(L)를 버리는 이유: 투명부는 배경 색을 보이되 플라스틱이 빛을 먹어 **어두워지므로**,
#   밝기까지 비교하면 본체와 구별이 안 된다. 게다가 케이스 자체의 요철 그늘·하이라이트가
#   밝기를 지배해 방향을 잃는다 — 실측에서 L 을 포함하자 왼쪽 음영에 속아 오답이 났다.
#   배경을 안 쓰는 것도 의도적이다: 그림자는 배경과 색이 같아 "투명"으로 오인되기 쉽다.
#
# 제거된 대안 (실측 근거):
#   - "배경 유사도"(bg): 띠가 케이스 바깥(바닥) 색 분포와 닮은 정도
#   - "이질도"(odd)   : 띠가 나머지 세 띠와 다른 정도
#   둘 다 히스토그램의 **피어슨 상관**으로 쟀는데, 이 지표는 두 분포가 겹치지 않으면
#   "얼마나 먼지"와 무관하게 비슷한 값으로 **포화**된다. 실측(분홍 케이스, 배경 RGB
#   130/129/128 · 힌지 103/94/96 · 본체 94/68/66)에서 네 변 점수가 0.15~0.18 안에 몰려
#   사실상 동점이 됐고, 정지 장면인데도 실행마다 답이 바뀌었다. 8회 반복 비교에서 bg 는
#   실제 제품(0/8)과 검은 케이스(3/8)에서 실패했고, odd 는 합성에선 통했지만 실물에선
#   설정 따라 답이 튀었다 — 실제 장면은 프레임 요철·인쇄 때문에 "튀는 변"이 여럿이라
#   "제일 튀는 변 = 힌지" 가정이 깨진다. 본체 색 대비는 세 상황 모두 정답이었고, 색이
#   없는 케이스(흰·검정)에서는 **낮은 신뢰도로 정직하게 보고**한다 — 틀린 답을 확신하는
#   것보다 낫다. 무채색 케이스가 실제로 등장하면 그 캡처로 전용 단서를 설계한다.


# chroma 방식의 신뢰도 환산: 1·2등 점수 차(Lab a*b* 단위)를 이 값으로 나눠 0~1 로 만든다.
# Lab 에서 1 단위는 겨우 구분되는 색차, 4~5 단위면 눈에 뚜렷한 차이 — 그 정도 격차면
# 확신해도 좋다는 뜻. (실측: 띠 두께를 실제 힌지 폭에 맞추면 격차 5.4, 3배 넓게 잡으면 1.1)
HINGE_CONF_LAB_SCALE = 4.0


def opening_from_hinge(
    mask,
    gray,
    obb,
    rgb=None,
    band_ratio: float = 0.25,
    debug: bool = False,
) -> Optional[Dict]:
    """힌지(투명한 변)를 네 변 중에서 골라, 그 **반대쪽**을 여는 방향으로 반환한다.

    정사각형 케이스처럼 경우의 수가 4가지인 경우를 위한 방식. 절차:
      1) OBB 로 케이스를 똑바로 세운다(warp). 마스크도 같이 warp 해 실제 객체 픽셀만 쓴다.
      2) 네 변 안쪽의 **띠(band)** 를 뜬다. 모서리는 두 띠에 겹치므로 잘라낸다
         (겹치면 인접 변끼리 점수가 섞여 판별력이 떨어진다).
      3) 띠마다 "투명함 점수"를 매긴다 (본체 색 대비 — 위 설명 참고). 점수 최대 = 힌지.
      4) 힌지의 **맞은편 변** 바깥 방향이 여는 방향. 원본 좌표계로 역투영해 반환.

    인자:
      rgb        : (선택) 컬러 이미지. 주면 채널별 히스토그램을 써서 판별력이 크게 오른다
                   — 투명부는 바닥 '색'까지 닮기 때문. 없으면 gray 만 사용.
      band_ratio : 변 안쪽 띠 두께 비율(0.05~0.45). 얇으면 노이즈, 두꺼우면 가운데
                   제품 영역이 섞여 변 사이 차이가 흐려진다.

    반환: {"dir", "angle_deg", "confidence", "half_len", "axis": "hinge",
           "hinge_side"(0~3), "scores"[4]} — 실패 시 None.
    confidence 는 1등과 2등 점수의 **절대 격차**(0~1)다. 4지선다에서는 "얼마나 확실히
    하나만 튀는가"가 곧 신뢰도이기 때문. **0.15 미만이면 네 변이 고만고만하다는 뜻**이고,
    가장 흔한 원인은 마스크가 투명한 힌지부를 포함하지 못해 네 띠가 모두 케이스 본체인
    경우다 (그러면 무엇을 골라도 사실상 찍는 것 — 디버그 뷰에서 막대 길이로 확인).
    """
    band_ratio = float(np.clip(band_ratio, 0.05, 0.45))
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    box = np.asarray(obb["box_pts"], np.float32)
    Wl = float(np.linalg.norm(box[1] - box[0]))
    Hl = float(np.linalg.norm(box[2] - box[1]))
    W, H = int(round(Wl)), int(round(Hl))
    if W < 16 or H < 16:
        return None

    # 1) 케이스를 똑바로 세운다 (canonical: 변0=위, 1=오른쪽, 2=아래, 3=왼쪽)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warp_mask = cv2.warpPerspective(mask_u8, M, (W, H), flags=cv2.INTER_NEAREST)
    # 디버그 화면에 띄울 warp 이미지 (점수 계산에는 아래 feat 를 쓴다)
    layers = [cv2.warpPerspective(np.asarray(gray), M, (W, H))]
    if rgb is not None:
        col = np.asarray(rgb)
        if col.ndim == 3 and col.shape[2] == 3:
            layers = [cv2.warpPerspective(col[:, :, c], M, (W, H)) for c in range(3)]

    # 2) 네 변 안쪽 띠 (모서리는 제외 — 두 변에 걸쳐 점수가 섞인다)
    tw = max(2, int(W * band_ratio))
    th = max(2, int(H * band_ratio))
    if W - 2 * tw < 4 or H - 2 * th < 4:
        return None
    slices = [
        (slice(0, th), slice(tw, W - tw)),  # 0: 위
        (slice(th, H - th), slice(W - tw, W)),  # 1: 오른쪽
        (slice(H - th, H), slice(tw, W - tw)),  # 2: 아래
        (slice(th, H - th), slice(0, tw)),  # 3: 왼쪽
    ]
    inner = (slice(th, H - th), slice(tw, W - tw))  # 띠를 뺀 안쪽 = 케이스 '본체' 표본

    # 3) 투명함 점수 = 색상면(Lab a*b*)에서 **본체 색으로부터의 거리** (높을수록 힌지)
    if rgb is not None:
        lab = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        feat = np.stack([cv2.warpPerspective(lab[:, :, c], M, (W, H)) for c in (1, 2)], axis=2)
    else:
        # 컬러가 없으면 밝기 편차로 대체 — 은색 힌지처럼 밝기가 다른 경우만 잡힌다.
        # 판별력이 크게 떨어지므로 호출부는 되도록 rgb 를 넘긴다.
        feat = cv2.warpPerspective(np.asarray(gray, dtype=np.uint8), M, (W, H)).astype(np.float32)[:, :, None]

    # 마스크 경계(배경과 섞인 픽셀·그림자 테두리)를 걷어낸다 — 안 걷으면 마스크가
    # 조금만 커져도 그쪽 띠가 배경색을 머금어 '투명'으로 오인된다. 다만 힌지 띠 자체가
    # 가장자리에 붙어 있어 많이 깎으면 신호까지 사라지므로 4% 로 제한한다.
    k = max(2, int(round(min(W, H) * 0.04)))
    # borderValue=0 이 **필수**: cv2.erode 의 기본 경계값은 최댓값이라 이미지 가장자리가
    # 깎이지 않는다. warp 마스크는 OBB 를 꽉 채우므로 기본값으로는 침식이 통째로 무효가
    # 되어(픽셀 수가 그대로), 마스크가 한쪽으로 삐져나와도 걸러내지 못했다.
    eroded = cv2.erode(
        warp_mask,
        np.ones((2 * k + 1, 2 * k + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if int(eroded.sum()) < 200:
        eroded = warp_mask  # 너무 작아지면 원본 유지 (작은 객체 안전장치)

    def _pix(sl):
        return feat[sl][eroded[sl] > 0]

    core = _pix(inner)
    if len(core) < 50:
        return None
    body = np.median(core, axis=0)  # 본체 색 = 안쪽 중앙값 (인쇄·요철에 덜 흔들리게 median)
    scores = []
    for sl in slices:
        p = _pix(sl)
        if len(p) < 30:  # 마스크가 OBB 모서리를 덜 채운 경우 등
            return None
        scores.append(float(np.linalg.norm(p - body, axis=1).mean()))

    order = int(np.argmax(scores))
    srt = sorted(scores, reverse=True)
    # 신뢰도 = 1등과 2등의 **절대** 격차를 Lab 색차 단위로 환산한 값.
    #   ※ 예전에는 전체 산포로 나눈 상대 격차를 썼는데, 네 변이 거의 동점일 때도
    #     (예: 힌지가 마스크 밖이라 신호가 없는 경우) 비율만 크면 conf 0.96 처럼 높게
    #     나와 "동점인데 확신하는" 위험이 있었다. 절대 격차는 그 상황을 제대로 0 에
    #     가깝게 떨어뜨린다.
    conf = float(np.clip((srt[0] - srt[1]) / HINGE_CONF_LAB_SCALE, 0.0, 1.0))

    # 4) 힌지의 맞은편 변 = 여는 쪽. canonical 중심 → 그 변 중점 방향을 원본으로 되돌린다
    opp = (order + 2) % 4
    mids = [(W / 2.0, 0.0), (W - 1.0, H / 2.0), (W / 2.0, H - 1.0), (0.0, H / 2.0)]
    Minv = cv2.getPerspectiveTransform(dst, box)
    pts = np.array([[[W / 2.0, H / 2.0]], [list(mids[opp])]], np.float32)
    o = cv2.perspectiveTransform(pts, Minv)
    d = o[1, 0] - o[0, 0]
    nd = float(np.linalg.norm(d))
    if nd < 1e-6:
        return None
    result = {
        "dir": (float(d[0] / nd), float(d[1] / nd)),
        "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
        "confidence": conf,
        "half_len": nd,
        "axis": "hinge",
        "hinge_side": order,
        "scores": [float(x) for x in scores],
    }
    if debug:
        debug_show_hinge(layers, warp_mask, slices, scores, order, conf)
    return result


def debug_show_hinge(layers, warp_mask, slices, scores, hinge, conf):
    """[개발용] 힌지 방식 중간 단계를 cv2.imshow 로 표시 — 어느 변이 왜 뽑혔는지 눈으로 확인.

    왼쪽: OBB 로 똑바로 세운 케이스(warp) 위에 네 띠를 그린 것.
          빨강 = 힌지로 고른 변, 초록 = 그 맞은편(여는 쪽), 회색 = 나머지.
          마스크 밖(OBB 모서리 중 객체가 아닌 부분)은 어둡게 — 점수 계산에서 빠진 영역.
    오른쪽: 네 변의 '투명함 점수' 막대. 1등과 2등의 격차가 곧 confidence 다.

    글자는 **ASCII 만** 쓴다 — cv2.putText 는 한글을 못 그려서 깨진다 (T/R/B/L =
    위/오른쪽/아래/왼쪽, canonical warp 기준). 다른 디버그 창과 같이 아무 키나
    누르면 다음 객체로 넘어간다.
    """
    try:
        # layers 는 호출부가 넘긴 rgb 의 채널(R,G,B) — cv2 는 BGR 로 그리므로 뒤집어 합친다.
        # (점수 계산은 채널 순서와 무관하지만, 디버그 화면 색이 실제와 달라 보이면 헷갈린다.)
        base = layers[0] if len(layers) == 1 else cv2.merge(layers[2::-1])
        vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base.copy()
        vis[warp_mask == 0] = (vis[warp_mask == 0] * 0.35).astype(vis.dtype)

        names = ["T", "R", "B", "L"]  # Top / Right / Bottom / Left (warp 기준)
        opp = (hinge + 2) % 4
        for i, (rs, cs) in enumerate(slices):
            color = (0, 0, 255) if i == hinge else ((0, 220, 0) if i == opp else (190, 190, 190))
            cv2.rectangle(vis, (cs.start, rs.start), (cs.stop - 1, rs.stop - 1), color, 1)
            cv2.putText(vis, names[i], (cs.start + 3, rs.start + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 작은 케이스도 읽히도록 확대 (다른 디버그 창과 같은 방식)
        scale = max(1, int(round(360 / max(vis.shape[0], 1))))
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # 점수 막대 — 상관값은 음수도 나오므로 [min,max] 로 정규화해 길이를 잡는다
        bh, bw = vis.shape[0], 260
        bars = np.full((bh, bw, 3), 30, np.uint8)
        lo, hi = float(min(scores)), float(max(scores))
        span = (hi - lo) or 1.0
        for i, sc in enumerate(scores):
            y = 46 + i * 34
            color = (0, 0, 255) if i == hinge else ((0, 220, 0) if i == opp else (190, 190, 190))
            cv2.rectangle(bars, (44, y), (44 + int((sc - lo) / span * 150), y + 18), color, -1)
            cv2.putText(bars, names[i], (8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(bars, f"{sc:+.3f}", (200, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.putText(bars, "score = dist. from body color", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(bars, f"conf={conf:.2f}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(bars, f"hinge={names[hinge]} -> open={names[opp]}", (8, bh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

        montage = cv2.hconcat([vis, np.zeros((bh, 6, 3), np.uint8), bars])
        win = "hinge debug: red=hinge, green=open (press any key -> next)"
        cv2.imshow(win, montage)
        cv2.waitKey(0)
        cv2.destroyWindow(win)
    except Exception as e:
        logger.warning(f"힌지 디버그 시각화 실패: {e}")
