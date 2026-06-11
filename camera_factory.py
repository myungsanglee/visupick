"""
Camera Factory
==============
카메라 종류 문자열 → BaseCamera 구현체 인스턴스.

새 카메라를 추가하려면:
  1. base_camera.BaseCamera 를 상속하는 클래스 작성
  2. _REGISTRY 에 ("이름", import 경로 + 클래스명) 등록
  3. main.py 의 카메라 선택 UI 콤보에 옵션 추가

지연 import:
  ZividCamera 와 RealSenseCamera 가 각각 무거운 SDK 의존성(zivid / pyrealsense2)을
  가지고 있어, 둘 다 시스템에 설치되어 있을 필요 없다. 사용하지 않는 카메라의
  SDK 가 빠져 있어도 다른 카메라로 본 프로그램을 실행할 수 있게, factory 가
  요청받은 카메라만 늦게 import.
"""

from typing import List, Tuple

from base_camera import BaseCamera


# 카메라 등록: (사람이 읽는 이름, 모듈 경로, 클래스명)
_REGISTRY: List[Tuple[str, str, str]] = [
    ("Zivid", "zivid_camera", "ZividCamera"),
    ("RealSense", "realsense_camera", "RealSenseCamera"),
    ("Percipio", "percipio_camera", "PercipioCamera"),
]


# 카메라별 설정 파일 다이얼로그 필터 (Zivid YML vs RealSense JSON 등)
# 새 카메라 추가 시 여기에도 등록.
_SETTINGS_FILTERS = {
    "Zivid": "Zivid YML 설정 (*.yml *.yaml)",
    "RealSense": "RealSense JSON 설정 (*.json)",
    "Percipio": "Percipio Feature 파일 (*.fea *.ini *.json)",
}


def get_settings_file_filter(name: str) -> str:
    """카메라 종류 → QFileDialog 파일 필터 문자열. 미등록은 'All Files (*)'."""
    return _SETTINGS_FILTERS.get(name.strip(), "All Files (*)")


def list_available_camera_names() -> List[str]:
    """카메라 선택 UI에 표시할 이름 목록."""
    return [name for name, _, _ in _REGISTRY]


def create_camera(name: str) -> BaseCamera:
    """이름으로 카메라 인스턴스 생성. SDK import 실패 시 RuntimeError.

    예: create_camera("Zivid"), create_camera("RealSense")
    """
    name_lower = name.strip().lower()
    for registered_name, module_path, class_name in _REGISTRY:
        if registered_name.lower() == name_lower:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                return cls()
            except Exception as e:
                raise RuntimeError(
                    f"카메라 '{registered_name}' 초기화 실패: {e}\n"
                    f"({module_path}.{class_name} 의 SDK 가 설치되어 있는지 확인.)"
                )
    raise ValueError(
        f"알 수 없는 카메라: '{name}'. 등록된 카메라: {list_available_camera_names()}"
    )
