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

from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


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
        size_changed = (
            self._bgr is None
            or bgr is None
            or self._bgr.shape != bgr.shape
        )
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
        M = np.array([
            [display, 0.0, offset_x],
            [0.0, display, offset_y],
        ], dtype=np.float32)
        warped = cv2.warpAffine(
            canvas, M, (label_w, label_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.BACKGROUND_BGR,
        )

        # 캐시 (_widget_to_image 와 _post_draw 가 사용)
        self._display_scale = display
        self._display_offset = (offset_x, offset_y)

        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data, label_w, label_h, label_w * 3,
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
