"""
VisionTabMixin — 캘리브레이션을 쓰는 탭 4개(빈 픽킹·CAD 매칭·표면 추적·검증)의 공통 기능.

캘리브레이션 상태는 main(VisuPickApp)의 CalibrationContext 하나가 소유하고
(calibration.py — 왜 한 곳인지 그쪽 docstring 참고), 탭은 이 믹스인의
T_calib/calib_mode **읽기 전용 프로퍼티**로 접근한다. 예전처럼 탭이
`self.T_calib = ...` 로 자기 복사본을 만들면 AttributeError 가 나도록 의도했다
— 상태가 두 곳에 살기 시작하면 다시 어긋난다.

`캘리브레이션 로드` 버튼도 여기 한 벌만 있다: 어느 탭에서 로드하든 main 이
모든 탭에 방송(_broadcast_calibration_loaded)해 라벨이 함께 갱신되고, 탭별
후처리는 _on_calibration_loaded() 훅으로 처리한다.

사용법: class XxxTab(VisionTabMixin, ..., QWidget) 으로 상속하고,
탭의 _init_ui 가 self.calib_label(QLabel) 을 만들어 두면 된다.
"""

import logging

from PySide6.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)


class VisionTabMixin:
    # ------------------------------------------------------------
    # 캘리브레이션 (읽기는 프로퍼티, 로드는 공용 한 벌)
    # ------------------------------------------------------------

    @property
    def T_calib(self):
        """4×4 cam 변환 (main.calib 소유 — 탭 로컬 복사본을 두지 않는다)."""
        return self.main.calib.T

    @property
    def calib_mode(self):
        return self.main.calib.mode

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "캘리브레이션 파일 선택",
            "data",
            "JSON Files (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            name = self.main.calib.load(path)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"로드 실패:\n{e}")
            return
        self.main.statusBar().showMessage(f"캘리브레이션 로드: {name} ({self.calib_mode}) — 모든 탭에 적용")
        self.main._broadcast_calibration_loaded(name)

    def _update_calib_label(self, name: str):
        """캘리브레이션 라벨 갱신 (방송으로 호출됨). 라벨 구성이 다른 탭은 오버라이드."""
        label = getattr(self, "calib_label", None)
        if label is not None:
            label.setText(f"{name} [{self.calib_mode}]")

    def _on_calibration_loaded(self):
        """캘리브레이션이 (어느 탭에서든) 로드된 뒤 탭별 후처리 훅. 기본은 없음."""
