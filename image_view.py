"""
ZoomableImageLabel — 줌(휠) / 팬(우클릭 드래그) 지원하는 QLabel 베이스.

5개 탭의 2D 이미지 뷰 (데이터 수집, 검증, Bin Picking, CAD 매칭, 표면 추적)가
공유하는 베이스. 자식 클래스는 다음 두 훅만 오버라이드하면 된다:

  - `_make_overlay_image()`  → np.ndarray (BGR)
      이미지 **원본 좌표계**에서 오버레이 (bbox, ROI, path, samples 등)를 그린
      캔버스를 반환. 줌/팬은 베이스가 알아서 적용한다.

  - `_post_draw(pixmap)`     → None
      **위젯 좌표계** 임시 오버레이 (드래그 중 점선 사각형 등). 필요할 때만.

좌클릭 / 이동 / 뗌 이벤트는 자식이 직접 오버라이드 — 베이스는 휠/우클릭만 가로채므로
기존 ROI 드래그·점 클릭·픽킹 로직과 충돌하지 않는다.

좌표 변환은 `_widget_to_image(QPoint)` 로 줌/팬을 반영해 호출하면 된다.
"""

from typing import Optional, Tuple, List

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel

# 객체 정보 라벨을 기준점(객체 중심 = 화살표 시작점)에서 띄우는 간격(px).
# 겹치지 않을 만큼만 작게 — 크면 라벨이 객체에서 떨어져 나가 어느 객체 것인지 흐려진다.
LABEL_OFFSET_X = 8
LABEL_OFFSET_Y = 8
# 화살표 시작점(객체 중심) 표시용 원의 반지름(px). 중심이 어디인지 한눈에 보이게.
CENTER_DOT_R = 5


class ZoomableImageLabel(QLabel):
    """줌·팬 가능한 이미지 뷰. 다음 오버라이드 훅으로 확장."""

    MIN_ZOOM = 0.2
    MAX_ZOOM = 16.0
    ZOOM_STEP = 1.25
    BACKGROUND_BGR = (40, 40, 40)  # 이미지가 위젯보다 작을 때 채울 색

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #2a2a2a; color: #888;")
        self.setMouseTracking(True)
        # 우클릭 컨텍스트 메뉴 등 막기 (Qt 기본 비활성이지만 명시)
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self._bgr: Optional[np.ndarray] = None

        # 사용자 뷰 상태
        self._zoom: float = 1.0
        self._pan: Tuple[float, float] = (0.0, 0.0)  # 위젯 좌표 픽셀 오프셋

        # 마지막 표시 변환 캐시 (위젯 → 이미지 좌표 변환용)
        self._display_scale: float = 1.0
        self._display_offset: Tuple[float, float] = (0.0, 0.0)

        # 우클릭 팬 상태
        self._panning: bool = False
        self._pan_start: Optional[QPoint] = None
        self._pan_start_value: Tuple[float, float] = (0.0, 0.0)

    # ============================================================
    # 외부 API
    # ============================================================

    def set_image(self, bgr: Optional[np.ndarray]):
        """원본 BGR 이미지 설정. 크기가 바뀌면 뷰가 자동 리셋된다."""
        size_changed = self._bgr is None or bgr is None or self._bgr.shape != bgr.shape
        self._bgr = bgr
        if size_changed:
            self._zoom = 1.0
            self._pan = (0.0, 0.0)
        self._refresh()

    def reset_view(self):
        """줌 1.0, 팬 0,0 (fit-to-window) 로 복귀."""
        self._zoom = 1.0
        self._pan = (0.0, 0.0)
        self._refresh()

    def clear_image(self):
        self._bgr = None
        self.setText("이미지 없음")
        self.setPixmap(QPixmap())

    @property
    def has_image(self) -> bool:
        return self._bgr is not None

    # ============================================================
    # 오버라이드 훅
    # ============================================================

    def _make_overlay_image(self) -> Optional[np.ndarray]:
        """
        이미지 **원본 좌표계** 캔버스 (BGR) 를 만들어 반환.
        bbox·ROI·path 같이 이미지에 묶인 오버레이는 모두 여기서 그린다.
        스케일 적용 X — 원본 픽셀 좌표 그대로 사용.

        기본 구현: 원본 이미지 사본.
        """
        return None if self._bgr is None else self._bgr.copy()

    def _post_draw(self, pixmap: QPixmap):
        """
        위젯 좌표계 임시 오버레이 (예: 드래그 중 점선 사각형)를
        `QPainter(pixmap)` 으로 그린다. 기본은 아무것도 안 함.
        """
        pass

    # ============================================================
    # 좌표 변환
    # ============================================================

    def _widget_to_image(self, pt: QPoint) -> Optional[Tuple[int, int]]:
        """위젯 좌표 → 이미지 원본 픽셀 좌표. 이미지 밖이면 None."""
        if self._bgr is None or self._display_scale <= 0:
            return None
        ox, oy = self._display_offset
        x = (pt.x() - ox) / self._display_scale
        y = (pt.y() - oy) / self._display_scale
        img_h, img_w = self._bgr.shape[:2]
        if not (0 <= x < img_w and 0 <= y < img_h):
            return None
        return int(x), int(y)

    # ============================================================
    # 표시
    # ============================================================

    def _refresh(self):
        if self._bgr is None:
            self.setText("이미지 없음")
            self.setPixmap(QPixmap())
            return

        canvas = self._make_overlay_image()
        if canvas is None:
            canvas = self._bgr.copy()

        img_h, img_w = canvas.shape[:2]
        label_w = self.width()
        label_h = self.height()
        if label_w <= 0 or label_h <= 0:
            return

        # fit-to-window scale × 사용자 줌
        fit = min(label_w / img_w, label_h / img_h)
        display = fit * self._zoom

        disp_w = img_w * display
        disp_h = img_h * display
        offset_x = (label_w - disp_w) / 2.0 + self._pan[0]
        offset_y = (label_h - disp_h) / 2.0 + self._pan[1]

        # 이미지 → 위젯 affine 변환 한 번에 적용 (메모리/시간 모두 효율)
        M = np.array(
            [
                [display, 0.0, offset_x],
                [0.0, display, offset_y],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            canvas,
            M,
            (label_w, label_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.BACKGROUND_BGR,
        )

        # 캐시 (_widget_to_image 와 _post_draw 가 사용)
        self._display_scale = display
        self._display_offset = (offset_x, offset_y)

        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            label_w,
            label_h,
            label_w * 3,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)

        # 자식의 위젯 좌표계 오버레이 (드래그 점선 등)
        self._post_draw(pixmap)

        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    # ============================================================
    # 휠 줌 (커서 위치 중심)
    # ============================================================

    def wheelEvent(self, event):
        if self._bgr is None:
            return
        cursor = event.position()
        cx, cy = float(cursor.x()), float(cursor.y())

        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return

        # 줌 전 커서 아래 있던 이미지 픽셀이 줌 후에도 같은 커서 위치에 있도록
        # 팬을 보정한다 (확대/축소가 커서 기준으로 자연스럽게 동작).
        old_img_x = (cx - self._display_offset[0]) / max(self._display_scale, 1e-9)
        old_img_y = (cy - self._display_offset[1]) / max(self._display_scale, 1e-9)

        self._zoom = new_zoom

        img_h, img_w = self._bgr.shape[:2]
        label_w = max(self.width(), 1)
        label_h = max(self.height(), 1)
        fit = min(label_w / img_w, label_h / img_h)
        new_display = fit * new_zoom
        base_ox = (label_w - img_w * new_display) / 2.0
        base_oy = (label_h - img_h * new_display) / 2.0
        # cursor = base_ox + pan_x + new_display * img_x
        # → pan_x = cursor - new_display * img_x - base_ox
        self._pan = (
            cx - new_display * old_img_x - base_ox,
            cy - new_display * old_img_y - base_oy,
        )
        self._refresh()
        event.accept()

    # ============================================================
    # 우클릭 드래그 = 팬, 우클릭 더블 = 뷰 리셋
    # ============================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._bgr is not None:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._pan_start_value = self._pan
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            cur = event.position().toPoint()
            dx = cur.x() - self._pan_start.x()
            dy = cur.y() - self._pan_start.y()
            self._pan = (
                self._pan_start_value[0] + dx,
                self._pan_start_value[1] + dy,
            )
            self._refresh()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.RightButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class DraggableImageLabel(ZoomableImageLabel):
    """마우스 드래그로 ROI 선택 + 클릭으로 객체 선택. 줌(휠)/팬(우클릭)은 베이스가 처리."""

    roiChanged = Signal(int, int, int, int)  # x1, y1, x2, y2 (이미지 원본 좌표)
    objectPicked = Signal(int)  # 클릭된 객체 인덱스

    CLICK_THRESHOLD = 8  # 위젯 좌표 픽셀 — 이하면 클릭, 초과면 드래그

    def __init__(self):
        super().__init__()
        # [(x1, y1, x2, y2, color, label, obj_index), ...]
        self._overlay_boxes: List[tuple] = []
        # [(mask(H,W) bool, color(BGR), obj_index), ...] — seg 모델일 때만 채워짐
        self._overlay_masks: List[tuple] = []
        # [(box_pts(4,2) int, color(BGR), obj_index, open_deg|None, open_conf|None), ...] — OBB 검출 시
        self._overlay_obbs: List[tuple] = []
        # [(start(x,y), end(x,y), color(BGR), obj_index), ...] — 여는 방향 화살표
        self._overlay_arrows: List[tuple] = []
        self._roi_rect: Optional[tuple] = None  # (x1, y1, x2, y2) 원본 이미지 좌표
        # Bin Box 를 이미지에 투영한 4점 폴리곤 (회전 포함). 있으면 _roi_rect 대신 이걸 그림.
        self._roi_poly: Optional[np.ndarray] = None
        self._highlighted_idx: Optional[int] = None

        # 좌클릭 드래그 상태 (위젯 좌표)
        self._dragging = False
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None

    # 기존 외부 API 호환 (이름·시그니처 유지)
    def set_boxes(self, boxes: List[tuple]):
        self._overlay_boxes = boxes
        self._refresh()

    def set_masks(self, masks: List[tuple]):
        """seg 모델 객체별 mask 오버레이. 빈 리스트면 아무것도 안 그림 (det 모델)."""
        self._overlay_masks = masks
        self._refresh()

    def set_obbs(self, obbs: List[tuple]):
        """OBB(회전 사각형) 오버레이. [(box_pts(4,2), color, obj_idx, open_deg, open_conf), ...].

        중심 원과 정보 라벨(`X: .., Y: ..`)은 항상 그린다. obj_idx 뒤의 두 항목은
        **여는 방향을 계산했을 때만** 넘기는 값으로, 있으면 라벨에 `Deg`(여는 방향 각도)
        와 `Open`(신뢰도)이 덧붙는다 — OBB 만 계산한 경우엔 좌표만 나온다.
        빈 리스트면 스킵.
        """
        self._overlay_obbs = obbs
        self._refresh()

    def set_arrows(self, arrows: List[tuple]):
        """여는 방향 화살표 오버레이. [(start(x,y), end(x,y), color, obj_idx), ...]. 빈 리스트면 스킵."""
        self._overlay_arrows = arrows
        self._refresh()

    @staticmethod
    def _compose_label(cx: float, cy: float, open_deg=None, open_conf=None) -> str:
        """정보 라벨 문구. 중심 픽셀 좌표는 항상, 여는 방향 각도·신뢰도는 **있을 때만**.

        Deg 는 **여는 방향 각도**다 (OBB 각도가 아님 — 그쪽은 minAreaRect 규약상 0~90° 로
        접혀 90° 마다 되돌아가고 경계에서 튀므로 방향 정보로 못 쓴다). 이미지 좌표계에서
        중심 → 여는 쪽 벡터의 atan2 이며, 0°=오른쪽 / +90°=아래 / ±180°=왼쪽 / −90°=위
        (이미지 y축이 아래를 향하므로 **화면상 시계방향이 +**). 범위 −180~180°.
        """
        label = f"X: {cx:.1f}, Y: {cy:.1f}"
        if open_deg is not None:
            label += f", Deg: {open_deg:.1f}"
        if open_conf is not None:
            label += f", Open: {open_conf:.2f}"
        return label

    @staticmethod
    def _draw_info_label(canvas, text: str, ax: int, ay: int, fill_color):
        """객체 정보 라벨 — 글자 길이에 맞춘 사각형을 fill_color 로 채우고 흰 글씨.

        기준점 (ax, ay) = 객체 중심(=화살표 시작점). 라벨의 **왼쪽 아래 모서리**를 그
        지점에서 살짝 오른쪽·위로 띄워 놓는다. 중심에 딱 맞추면 화살표 시작점과 겹쳐
        글씨를 가리기 때문 — 라벨이 중심 오른쪽 위에 떠 있으면 화살표가 어느 방향으로
        나가든(아래/왼쪽이 흔함) 시작점이 보인다.

        라벨이 길어(X/Y/Deg/Open) 가장자리 객체에서는 화면 밖으로 나갈 수 있으므로
        캔버스 안으로 밀어 넣는다(잘려서 안 보이는 것보다 살짝 어긋나는 편이 낫다).
        """
        font, scale, th_px = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), base = cv2.getTextSize(text, font, scale, th_px)
        pad = 4
        bw, bh = tw + 2 * pad, th + base + 2 * pad
        x0 = ax + LABEL_OFFSET_X  # 왼쪽 모서리를 기준점 오른쪽으로
        y0 = ay - LABEL_OFFSET_Y - bh  # 아래 모서리를 기준점 위로 → top = 아래 − 높이
        H, W = canvas.shape[:2]
        x0 = max(0, min(x0, W - bw))
        y0 = max(0, min(y0, H - bh))
        fill = tuple(int(c) for c in fill_color)
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), fill, -1)
        cv2.putText(canvas, text, (x0 + pad, y0 + pad + th), font, scale, (255, 255, 255), th_px, cv2.LINE_AA)

    def set_highlight(self, idx: Optional[int]):
        self._highlighted_idx = idx
        self._refresh()

    def set_roi(self, rect: Optional[tuple]):
        self._roi_rect = rect
        self._refresh()

    def set_roi_polygon(self, pts):
        """회전된 ROI(=Bin Box 투영) 표시용 4점 폴리곤. None 이면 해제.

        2D 드래그는 축 정렬 사각형만 만들지만, Bin Box 는 base 좌표계에서 yaw 를 가질 수
        있어 이미지에 투영하면 기울어진 사각형이 된다. 그걸 그대로 보여줘야 실제 작업
        볼륨과 화면이 일치한다.
        """
        self._roi_poly = None if pts is None else np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        self._refresh()

    def clear_all(self):
        self._overlay_boxes = []
        self._overlay_masks = []
        self._overlay_obbs = []
        self._overlay_arrows = []
        self._roi_rect = None
        self._roi_poly = None
        self._highlighted_idx = None
        self.clear_image()

    # 이미지 좌표계 오버레이 (bbox, ROI 등)
    def _make_overlay_image(self) -> Optional[np.ndarray]:
        if self._bgr is None:
            return None
        canvas = self._bgr.copy()

        # seg mask 반투명 오버레이 (bbox 아래 레이어). det 모델이면 빈 리스트 → 스킵.
        for m_item in self._overlay_masks:
            mask, color = m_item[0], m_item[1]
            obj_idx = m_item[2] if len(m_item) > 2 else None
            if mask is None or mask.shape[:2] != canvas.shape[:2] or not mask.any():
                continue

            if self._highlighted_idx is not None and obj_idx == self._highlighted_idx:
                fill, alpha = (0, 255, 0), 0.55  # 선택된 객체: 초록 진하게
            elif self._highlighted_idx is not None:
                fill, alpha = tuple(int(c * 0.5) for c in color), 0.25  # 나머지 흐리게
            else:
                fill, alpha = color, 0.45

            fill_arr = np.array(fill, dtype=np.float32)
            region = canvas[mask].astype(np.float32)
            canvas[mask] = (region * (1.0 - alpha) + fill_arr * alpha).astype(np.uint8)

        for box in self._overlay_boxes:
            x1, y1, x2, y2, color, label = box[:6]
            obj_idx = box[6] if len(box) > 6 else None

            if self._highlighted_idx is not None and obj_idx == self._highlighted_idx:
                box_color = (0, 255, 0)
                thickness = 3
            elif self._highlighted_idx is not None:
                box_color = tuple(int(c * 0.5) for c in color)
                thickness = 2
            else:
                box_color = color
                thickness = 2

            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), box_color, thickness)

            num_str = f"#{obj_idx + 1}" if obj_idx is not None else ""
            full_label = f"{num_str} {label}".strip() if label else num_str
            if full_label:
                (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(iy1 - 4, th + 2)
                cv2.rectangle(canvas, (ix1, ty - th - 4), (ix1 + tw + 4, ty + 2), box_color, -1)
                cv2.putText(canvas, full_label, (ix1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # OBB(회전 사각형) — bbox 위 레이어. minAreaRect 결과.
        # 중심 원·정보 라벨은 화살표까지 다 그린 **뒤** 최상단에 올린다 (화살표에 가려지지 않게)
        pending_dots = []  # [(cx, cy, color), ...]
        pending_labels = []
        for obb in self._overlay_obbs:
            box_pts, color = obb[0], obb[1]
            obj_idx = obb[2] if len(obb) > 2 else None
            # 아래 둘은 **여는 방향을 계산했을 때만** 채워진다 (OBB 만 계산하면 None → 라벨에서 생략).
            open_deg = obb[3] if len(obb) > 3 else None  # 여는 방향 각도(이미지 좌표, atan2)
            open_conf = obb[4] if len(obb) > 4 else None  # 여는 방향 신뢰도
            if self._highlighted_idx is not None and obj_idx == self._highlighted_idx:
                obb_color, thickness = (0, 255, 0), 3
            elif self._highlighted_idx is not None:
                obb_color, thickness = tuple(int(c * 0.5) for c in color), 2
            else:
                obb_color, thickness = color, 2
            pts = np.asarray(box_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=obb_color, thickness=thickness)
            # 정보 라벨: 픽셀 좌표(항상) + 여는 방향 각도·신뢰도(계산했을 때만).
            # 글자 길이에 맞춘 사각형을 **객체 색으로 채우고 흰 글씨** — 객체가 여럿일 때
            # 어느 라벨이 어느 객체 것인지 색으로 바로 매칭되게 (bbox/마스크와 같은 색).
            cx = int(np.mean(pts[:, 0, 0]))
            cy = int(np.mean(pts[:, 0, 1]))
            label = self._compose_label(cx, cy, open_deg, open_conf)
            # 화살표보다 **나중에** 그려야 화살표가 글씨를 덮지 않는다 → 여기선 모으기만
            pending_labels.append((label, cx, cy, obb_color))
            # OBB 만 계산한 경우(여는 방향 없음)에도 중심이 보이도록 여기서 원을 모은다
            pending_dots.append((cx, cy, obb_color))

        # 여는 방향 화살표 — 중심 → 여는 쪽(뚜껑 열림). OBB 위 최상단 레이어.
        for arr in self._overlay_arrows:
            start, end, color = arr[0], arr[1], arr[2]
            obj_idx = arr[3] if len(arr) > 3 else None
            # arr[4](여는 방향 신뢰도)는 중심 정보 라벨에서 표시하므로 여기선 안 쓴다
            if self._highlighted_idx is not None and obj_idx == self._highlighted_idx:
                arr_color, thickness = (0, 255, 0), 3
            elif self._highlighted_idx is not None:
                arr_color, thickness = tuple(int(c * 0.5) for c in color), 2
            else:
                arr_color, thickness = color, 3
            p0 = (int(round(start[0])), int(round(start[1])))
            p1 = (int(round(end[0])), int(round(end[1])))
            cv2.arrowedLine(canvas, p0, p1, arr_color, thickness, tipLength=0.25)
            # 시작점(= 객체 중심) 원. OBB 가 함께 있으면 위에서 이미 같은 점을 모았으므로
            # 아래 dedupe 가 걸러낸다 (OBB 없이 화살표만 주는 호출도 지원하려고 여기서도 모음).
            pending_dots.append((p0[0], p0[1], arr_color))
            # 여는 방향 신뢰도는 촉이 아니라 **정보 라벨**(Open:)에 함께 표시한다 —
            # 화살표가 짧으면 촉 라벨이 정보 라벨과 겹쳐 읽기 어려웠다.

        # 중심 원 — 화살표 **뒤에** 그려 꼬리에 가려지지 않게. 흰 테두리는 어두운 배경이나
        # 같은 색 객체 위에서도 원이 묻히지 않게 하는 용도 (1px 는 안티앨리어싱에 묻힘).
        drawn = set()
        for dx, dy, dcolor in pending_dots:
            if (dx, dy) in drawn:  # OBB·화살표가 같은 중심을 가리키는 경우 한 번만
                continue
            drawn.add((dx, dy))
            cv2.circle(canvas, (dx, dy), CENTER_DOT_R + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (dx, dy), CENTER_DOT_R, tuple(int(c) for c in dcolor), -1, cv2.LINE_AA)

        # 정보 라벨 — 최상단 (원 위에 올라가도 오프셋 때문에 겹치지 않는다)
        for text, lx, ly, lcolor in pending_labels:
            self._draw_info_label(canvas, text, lx, ly, lcolor)

        if self._roi_poly is not None and len(self._roi_poly) >= 3:
            # Bin Box 투영 폴리곤 (회전 반영) — 축 정렬 사각형보다 우선
            ip = np.asarray(self._roi_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [ip], isClosed=True, color=(0, 255, 255), thickness=2)
            tx, ty = int(self._roi_poly[:, 0].min()), int(self._roi_poly[:, 1].min())
            cv2.putText(canvas, "BIN", (tx, max(ty - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif self._roi_rect is not None:
            rx1, ry1, rx2, ry2 = self._roi_rect
            cv2.rectangle(canvas, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 255), 2)
            cv2.putText(canvas, "ROI", (int(rx1), max(int(ry1) - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return canvas

    # 위젯 좌표계 임시 오버레이 (드래그 중 점선 사각형)
    def _post_draw(self, pixmap):
        if not (self._dragging and self._drag_start and self._drag_current):
            return
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())
        painter.end()

    # 좌클릭만 처리 — 우클릭은 베이스 (팬) 에 위임
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._bgr is not None:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_current = event.position().toPoint()
            self._refresh()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._dragging:
            super().mouseReleaseEvent(event)
            return

        self._dragging = False
        end = event.position().toPoint()
        start_pt = self._drag_start
        self._drag_start = None
        self._drag_current = None
        if start_pt is None:
            self._refresh()
            return

        dx = end.x() - start_pt.x()
        dy = end.y() - start_pt.y()
        dist = (dx * dx + dy * dy) ** 0.5

        if dist <= self.CLICK_THRESHOLD:
            # 클릭으로 처리 → bbox 픽킹
            click_img = self._widget_to_image(end)
            if click_img and self._overlay_boxes:
                cx, cy = click_img
                hit_idx = None
                hit_area = float("inf")
                for box in self._overlay_boxes:
                    x1, y1, x2, y2 = box[:4]
                    obj_idx = box[6] if len(box) > 6 else None
                    if obj_idx is None:
                        continue
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < hit_area:
                            hit_area = area
                            hit_idx = obj_idx
                if hit_idx is not None:
                    self.objectPicked.emit(hit_idx)
            self._refresh()
            return

        # 드래그 → ROI 설정
        start_img = self._widget_to_image(start_pt)
        end_img = self._widget_to_image(end)
        if start_img and end_img:
            x1, y1 = start_img
            x2, y2 = end_img
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self._roi_rect = (x1, y1, x2, y2)
                self.roiChanged.emit(x1, y1, x2, y2)
        self._refresh()
