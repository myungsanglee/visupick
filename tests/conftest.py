"""pytest 공용 설정 — Qt 위젯을 쓰는 테스트를 위한 오프스크린 QApplication.

위젯 렌더(오버레이 그리기 순서 등)는 순수 함수로는 검증할 수 없어 실제 위젯이
필요하다. 디스플레이 없이 돌도록 QT_QPA_PLATFORM=offscreen 을 강제한다.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
