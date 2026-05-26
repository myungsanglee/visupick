"""
KUKA Robot C3Bridge Communication Module
C3Bridge 프로토콜(KukaVarProxy 호환)을 이용한 KUKA 로봇 제어
포트: 7000 (TCP), 바이너리 프로토콜
"""

import socket
import struct
import re
import time
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_robot_mode(raw: Optional[str]) -> str:
    """
    KUKA $MODE_OP 응답을 정규화한다.

    실제 반환 예: '#T1', '#T2', '#AUT', '#EX' (앞 # / 공백 / 대소문자 제각각).
    → 'T1', 'T2', 'AUT', 'EX' 형태로 통일.
    """
    if not raw:
        return "?"
    return raw.strip().upper().replace("#", "")


def is_auto_mode(mode: str) -> bool:
    """
    자동 운용 모드(AUT 또는 EXT) 여부.

    두 탭이 제각기 substring/exact 매칭을 쓰던 것을 하나로 통일.
    AUT, AUT_EXT, EX, EXT 모두 자동으로 간주 (안전하게 넓게).
    """
    m = (mode or "").upper()
    return "AUT" in m or m.startswith("EX")


class C3BridgeClient:
    """
    C3Bridge / KukaVarProxy 프로토콜 클라이언트

    프로토콜 포맷:
        요청: Tag(2B) + MsgLength(2B) + MsgType(1B) + Payload
        응답: Tag(2B) + MsgLength(2B) + MsgType(1B) + Payload + ErrorCode(2B) + Success(1B)

    Message Types:
        0 = Read Variable (ASCII)
        1 = Write Variable (ASCII)
        11 = Motion Control
    """

    # Message Types
    MSG_READ = 0
    MSG_WRITE = 1
    MSG_MOTION = 11

    # Motion Types
    MOTION_PTP = 1
    MOTION_LIN = 2
    MOTION_PTP_REL = 3
    MOTION_LIN_REL = 4

    def __init__(self, host: str, port: int = 7000, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self._tag_counter = 0

    def _next_tag(self) -> int:
        self._tag_counter = (self._tag_counter + 1) % 65536
        return self._tag_counter

    def connect(self) -> bool:
        """C3Bridge 서버에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"C3Bridge 연결 성공: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"C3Bridge 연결 실패: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """연결 해제"""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.connected = False
            logger.info("C3Bridge 연결 해제")

    def _send_and_receive(self, data: bytes) -> bytes:
        """바이너리 메시지 송수신"""
        self.socket.sendall(data)

        # 헤더 수신 (Tag 2B + Length 2B)
        header = self._recv_exact(4)
        tag, msg_len = struct.unpack(">HH", header)

        # 페이로드 수신
        payload = self._recv_exact(msg_len)
        return header + payload

    def _recv_exact(self, n: int) -> bytes:
        """정확히 n바이트 수신"""
        data = b""
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("연결이 끊어졌습니다")
            data += chunk
        return data

    def read_variable(self, var_name: str) -> Optional[str]:
        """
        KRL 변수 읽기 (Message Type 0)

        요청 포맷: Tag(2B) + Length(2B) + Type(1B=0) + NameLen(2B) + Name(ASCII)
        응답 포맷: Tag(2B) + Length(2B) + Type(1B=0) + ValueLen(2B) + Value(ASCII) + ErrCode(2B) + Success(1B)

        Args:
            var_name: KRL 변수명 (예: "$POS_ACT", "$AXIS_ACT")

        Returns:
            변수 값 문자열 또는 None
        """
        if not self.connected:
            logger.error("연결되어 있지 않습니다")
            return None

        try:
            tag = self._next_tag()
            name_bytes = var_name.encode("ascii")
            name_len = len(name_bytes)

            # 페이로드: Type(1B) + NameLen(2B) + Name
            payload = struct.pack(">BH", self.MSG_READ, name_len) + name_bytes
            msg_len = len(payload)

            # 전체 메시지: Tag(2B) + MsgLen(2B) + Payload
            message = struct.pack(">HH", tag, msg_len) + payload

            response = self._send_and_receive(message)

            # 응답 파싱
            resp_tag, resp_len = struct.unpack(">HH", response[:4])
            resp_body = response[4:]

            msg_type = resp_body[0]
            value_len = struct.unpack(">H", resp_body[1:3])[0]
            value = resp_body[3 : 3 + value_len].decode("ascii")

            # 마지막 3바이트: ErrorCode(2B) + Success(1B)
            err_code = struct.unpack(">H", resp_body[3 + value_len : 5 + value_len])[0]
            success = resp_body[5 + value_len]

            if success:
                logger.info(f"읽기 성공: {var_name} = {value}")
                return value
            else:
                logger.error(f"읽기 실패: {var_name}, 에러코드: {err_code}, 응답 원본: {resp_body.hex()}")
                return None

        except Exception as e:
            logger.error(f"변수 읽기 오류 ({var_name}): {e}")
            return None

    def write_variable(self, var_name: str, value: str) -> bool:
        """
        KRL 변수 쓰기 (Message Type 1)

        요청 포맷: Tag(2B) + Length(2B) + Type(1B=1) + NameLen(2B) + Name + ValueLen(2B) + Value
        응답 포맷: Tag(2B) + Length(2B) + Type(1B=1) + ValueLen(2B) + Value + ErrCode(2B) + Success(1B)

        Args:
            var_name: KRL 변수명
            value: 설정할 값 (문자열)

        Returns:
            성공 여부
        """
        if not self.connected:
            logger.error("연결되어 있지 않습니다")
            return False

        try:
            tag = self._next_tag()
            name_bytes = var_name.encode("ascii")
            value_bytes = value.encode("ascii")

            # 페이로드: Type(1B) + NameLen(2B) + Name + ValueLen(2B) + Value
            payload = struct.pack(">BH", self.MSG_WRITE, len(name_bytes)) + name_bytes + struct.pack(">H", len(value_bytes)) + value_bytes
            msg_len = len(payload)

            message = struct.pack(">HH", tag, msg_len) + payload
            response = self._send_and_receive(message)

            # 응답 파싱
            resp_body = response[4:]
            msg_type = resp_body[0]
            val_len = struct.unpack(">H", resp_body[1:3])[0]
            err_code = struct.unpack(">H", resp_body[3 + val_len : 5 + val_len])[0]
            success = resp_body[5 + val_len]

            if success:
                logger.info(f"쓰기 성공: {var_name} = {value}")
                return True
            else:
                logger.error(f"쓰기 실패: {var_name}, 에러코드: {err_code}")
                return False

        except Exception as e:
            logger.error(f"변수 쓰기 오류 ({var_name}): {e}")
            return False

    def send_motion(self, motion_type: int, position_str: str) -> bool:
        """
        로봇 이동 명령 (Message Type 11)

        요청 포맷: Tag(2B) + Length(2B) + Type(1B=11) + MotionType(1B) + PosStrLen(2B) + PosStr(UTF-16LE)
        응답 포맷: Tag(2B) + Length(2B) + Type(1B=11) + ErrCode(2B) + Success(1B)

        Args:
            motion_type: 이동 타입 (1=PTP, 2=LIN, 3=PTP_REL, 4=LIN_REL)
            position_str: KRL 위치 문자열 (예: "{X 100, Y 200, Z 300, A 0, B 0, C 0}")

        Returns:
            성공 여부
        """
        if not self.connected:
            logger.error("연결되어 있지 않습니다")
            return False

        try:
            tag = self._next_tag()
            pos_bytes = position_str.encode("utf-16-le")

            # 페이로드: Type(1B) + MotionType(1B) + PosStrLen(2B) + PosStr(UTF-16LE)
            payload = struct.pack(">BBH", self.MSG_MOTION, motion_type, len(pos_bytes)) + pos_bytes
            message = struct.pack(">HH", tag, len(payload)) + payload

            response = self._send_and_receive(message)

            resp_body = response[4:]
            err_code = struct.unpack(">H", resp_body[1:3])[0]
            success = resp_body[3]

            if success:
                logger.info(f"이동 명령 성공: type={motion_type}, pos={position_str}")
                return True
            else:
                logger.error(f"이동 명령 실패: 에러코드={err_code}")
                return False

        except Exception as e:
            logger.error(f"이동 명령 오류: {e}")
            return False


class KUKARobot:
    """KUKA 로봇 제어 고수준 인터페이스"""

    def __init__(self, host: str, port: int = 7000, tool_num: int = 0, base_num: int = 0):
        """
        Args:
            host: 로봇 IP 주소
            port: C3Bridge 포트 (기본 7000)
            tool_num: 사용할 Tool 번호 (TOOL_DATA[n])
            base_num: 사용할 Base 번호 (0=월드좌표, 1~=BASE_DATA[n])
        """
        self.client = C3BridgeClient(host, port)
        self.tool_num = tool_num
        self.base_num = base_num

    def connect(self) -> bool:
        if not self.client.connect():
            return False
        self._setup_tool_base()
        return True

    def _setup_tool_base(self):
        """Tool/Base 데이터를 $TOOL, $BASE에 설정"""
        # Tool 설정 (0 = $NULLFRAME = 플랜지 중심)
        if self.tool_num == 0:
            tool_data = "$NULLFRAME"
        else:
            tool_data = self.client.read_variable(f"TOOL_DATA[{self.tool_num}]")
        if tool_data:
            self.client.write_variable("$TOOL", tool_data)
            logger.info(f"Tool 설정 완료: Tool {self.tool_num} → {tool_data}")

        # Base 설정 (0 = $NULLFRAME = 월드 좌표계)
        if self.base_num == 0:
            base_frame = "$NULLFRAME"
        else:
            base_frame = self.client.read_variable(f"BASE_DATA[{self.base_num}]")
        if base_frame:
            self.client.write_variable("$BASE", base_frame)
            logger.info(f"Base 설정 완료: Base {self.base_num} → {base_frame}")

    def disconnect(self):
        self.client.disconnect()

    @staticmethod
    def _parse_krl_position(pos_str: str) -> Optional[Dict[str, float]]:
        """
        KRL 위치 문자열 파싱

        예: "{X 100.0, Y 200.0, Z 300.0, A 0.0, B 0.0, C 0.0}"
        → {"x": 100.0, "y": 200.0, "z": 300.0, "a": 0.0, "b": 0.0, "c": 0.0}
        """
        if not pos_str:
            return None

        result = {}
        # {KEY value, KEY value, ...} 형식 파싱
        pattern = r"([A-Za-z]\d*)\s+([-+]?\d*\.?\d+)"
        matches = re.findall(pattern, pos_str)
        for key, val in matches:
            result[key.lower()] = float(val)

        return result if result else None

    @staticmethod
    def _format_krl_position(pos: Dict[str, float]) -> str:
        """
        Dict를 KRL 위치 문자열로 변환

        예: {"x": 100, "y": 200, "z": 300, "a": 0, "b": 0, "c": 0}
        → "{X 100.0, Y 200.0, Z 300.0, A 0.0, B 0.0, C 0.0}"
        """
        parts = []
        for key in ["x", "y", "z", "a", "b", "c"]:
            if key in pos:
                parts.append(f"{key.upper()} {pos[key]:.4f}")
        return "{" + ", ".join(parts) + "}"

    def get_tcp_position(self) -> Optional[Dict[str, float]]:
        """
        현재 TCP(Tool Center Point) 위치 가져오기

        KRL 변수 $POS_ACT 읽기
        Returns: {"x", "y", "z", "a", "b", "c"} (mm, 도)
        """
        raw = self.client.read_variable("$POS_ACT")
        if raw:
            return self._parse_krl_position(raw)
        return None

    def get_axis_position(self) -> Optional[Dict[str, float]]:
        """
        현재 조인트(축) 각도 가져오기

        KRL 변수 $AXIS_ACT 읽기
        Returns: {"a1", "a2", "a3", "a4", "a5", "a6"} (도)
        """
        raw = self.client.read_variable("$AXIS_ACT")
        if raw:
            return self._parse_krl_position(raw)
        return None

    def get_tool_data(self) -> Optional[Dict[str, float]]:
        """현재 Tool 데이터 읽기 ($TOOL)"""
        raw = self.client.read_variable("$TOOL")
        if raw:
            return self._parse_krl_position(raw)
        return None

    def get_base_data(self) -> Optional[Dict[str, float]]:
        """현재 Base 데이터 읽기 ($BASE)"""
        raw = self.client.read_variable("$BASE")
        if raw:
            return self._parse_krl_position(raw)
        return None

    # ==========================================================
    # 큐 기반 이동 제어 (ext_move.src 20개 모션 큐)
    # ==========================================================
    # 동작:
    #   1. 다음 빈 슬롯(robo_motion_type[i]==0)에 목표 위치/타입 쓰기
    #   2. KRL 프로그램이 자동으로 순차 실행 (index 증가)
    #   3. 실행 완료 시 robo_motion_type[i] = 0으로 리셋됨
    # 장점: 연속 명령을 큐에 쌓아놓고 로봇이 알아서 처리
    # ==========================================================

    # 모션 타입 상수
    MT_IDLE = 0
    MT_PTP = 1
    MT_PTP_REL = 2
    MT_LIN = 3
    MT_LIN_REL = 4

    QUEUE_SIZE = 20

    def _find_empty_slot(self, timeout: float = 5.0) -> Optional[int]:
        """
        현재 robo_motion_index부터 순환하며 빈 슬롯(type=0) 찾기.

        KRL은 robo_motion_index가 가리키는 슬롯의 motion_type이 0이 아닐 때까지
        대기하므로, Python은 그 슬롯(또는 그 직후 슬롯)에 채워야 즉시 실행됨.
        """
        start = time.time()
        while time.time() - start < timeout:
            # 현재 KRL이 처리 중인 인덱스부터 시작 (1~20)
            try:
                idx_str = self.client.read_variable("robo_motion_index")
                cur_idx = int(idx_str.strip()) if idx_str else 1
                if cur_idx < 1 or cur_idx > self.QUEUE_SIZE:
                    cur_idx = 1
            except (ValueError, AttributeError):
                cur_idx = 1

            # cur_idx부터 시작해 순환 검색
            for offset in range(self.QUEUE_SIZE):
                slot = ((cur_idx - 1 + offset) % self.QUEUE_SIZE) + 1
                val = self.client.read_variable(f"robo_motion_type[{slot}]")
                if val is not None and val.strip() == "0":
                    return slot
            time.sleep(0.1)
        return None

    def _enqueue_motion(
        self,
        motion_type: int,
        use_e6pos: bool,
        pos: Dict[str, float],
        timeout: float = 5.0,
    ) -> Optional[int]:
        """
        모션을 큐에 추가.

        Returns:
            슬롯 번호 (1~20) on success, None on failure
        """
        slot = self._find_empty_slot(timeout)
        if slot is None:
            logger.error(f"큐에 빈 슬롯 없음 ({self.QUEUE_SIZE}개 모두 사용 중)")
            return None

        # 좌표 쓰기
        if use_e6pos:
            # robo_E6POS[]는 FRAME 타입 (X, Y, Z, A, B, C만, S/T 없음)
            pos_str = f"{{FRAME: X {pos['x']:.4f}, Y {pos['y']:.4f}, Z {pos['z']:.4f}, " f"A {pos['a']:.4f}, B {pos['b']:.4f}, C {pos['c']:.4f}}}"
            if not self.client.write_variable(f"robo_E6POS[{slot}]", pos_str):
                return None
            if not self.client.write_variable(f"robo_motion_mode[{slot}]", "TRUE"):
                return None
        else:
            # E6AXIS는 6축 + 6 외부축. 외부축 없으면 0.0 처리
            pos_str = (
                f"{{E6AXIS: A1 {pos['a1']:.4f}, A2 {pos['a2']:.4f}, A3 {pos['a3']:.4f}, "
                f"A4 {pos['a4']:.4f}, A5 {pos['a5']:.4f}, A6 {pos['a6']:.4f}, "
                f"E1 0.0, E2 0.0, E3 0.0, E4 0.0, E5 0.0, E6 0.0}}"
            )
            if not self.client.write_variable(f"robo_E6AXIS[{slot}]", pos_str):
                return None
            if not self.client.write_variable(f"robo_motion_mode[{slot}]", "FALSE"):
                return None

        # motion_type 마지막에 써야 KRL이 실행 시작 (0이 아닌 값)
        if not self.client.write_variable(f"robo_motion_type[{slot}]", str(motion_type)):
            return None

        logger.info(f"모션 큐 추가: slot={slot}, type={motion_type}, mode={'E6POS' if use_e6pos else 'AXIS'}")
        return slot

    def _wait_slot_done(self, slot: int, timeout: float = 60.0) -> bool:
        """특정 슬롯의 모션이 완료될 때까지 대기 (type==0이면 완료)"""
        start = time.time()
        while time.time() - start < timeout:
            val = self.client.read_variable(f"robo_motion_type[{slot}]")
            if val is not None and val.strip() == "0":
                return True
            time.sleep(0.1)
        return False

    def wait_queue_empty(self, timeout: float = 120.0) -> bool:
        """큐의 모든 모션이 완료될 때까지 대기"""
        start = time.time()
        while time.time() - start < timeout:
            all_done = True
            for i in range(1, self.QUEUE_SIZE + 1):
                val = self.client.read_variable(f"robo_motion_type[{i}]")
                if val is not None and val.strip() != "0":
                    all_done = False
                    break
            if all_done:
                return True
            time.sleep(0.2)
        return False

    # ---- 고수준 이동 메서드 (큐에 추가 후 즉시 반환, 슬롯 번호 반환) ----

    def add_move_ptp(self, x: float, y: float, z: float, a: float = 0, b: float = 0, c: float = 0) -> Optional[int]:
        """PTP 이동을 큐에 추가 (절대 좌표). S/T는 KUKA 자동 선택. 슬롯 번호 반환"""
        return self._enqueue_motion(
            self.MT_PTP,
            True,
            {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c},
        )

    def add_move_lin(self, x: float, y: float, z: float, a: float = 0, b: float = 0, c: float = 0) -> Optional[int]:
        """LIN 이동을 큐에 추가 (절대 좌표). S/T는 KUKA 자동 선택. 슬롯 번호 반환"""
        return self._enqueue_motion(
            self.MT_LIN,
            True,
            {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c},
        )

    def add_move_ptp_rel(self, dx: float = 0, dy: float = 0, dz: float = 0, da: float = 0, db: float = 0, dc: float = 0) -> Optional[int]:
        """PTP 상대 이동 큐에 추가. 슬롯 번호 반환"""
        return self._enqueue_motion(
            self.MT_PTP_REL,
            True,
            {"x": dx, "y": dy, "z": dz, "a": da, "b": db, "c": dc},
        )

    def add_move_lin_rel(self, dx: float = 0, dy: float = 0, dz: float = 0, da: float = 0, db: float = 0, dc: float = 0) -> Optional[int]:
        """LIN 상대 이동 큐에 추가. 슬롯 번호 반환"""
        return self._enqueue_motion(
            self.MT_LIN_REL,
            True,
            {"x": dx, "y": dy, "z": dz, "a": da, "b": db, "c": dc},
        )

    def add_move_axis(self, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float) -> Optional[int]:
        """축 좌표계 PTP 이동 큐에 추가. 슬롯 번호 반환"""
        return self._enqueue_motion(
            self.MT_PTP,
            False,
            {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6},
        )

    # ---- 블로킹 이동 (큐에 추가 후 해당 슬롯 완료까지 대기) ----

    def move_ptp(self, x: float, y: float, z: float, a: float = 0, b: float = 0, c: float = 0, timeout: float = 60.0) -> bool:
        """PTP 이동 (완료까지 블로킹)"""
        slot = self.add_move_ptp(x, y, z, a, b, c)
        if slot is None:
            return False
        return self._wait_slot_done(slot, timeout=timeout)

    def move_lin(self, x: float, y: float, z: float, a: float = 0, b: float = 0, c: float = 0, timeout: float = 60.0) -> bool:
        """LIN 이동 (완료까지 블로킹)"""
        slot = self.add_move_lin(x, y, z, a, b, c)
        if slot is None:
            return False
        return self._wait_slot_done(slot, timeout=timeout)

    def move_ptp_rel(self, dx: float = 0, dy: float = 0, dz: float = 0, da: float = 0, db: float = 0, dc: float = 0, timeout: float = 60.0) -> bool:
        """PTP 상대 이동 (완료까지 블로킹)"""
        slot = self.add_move_ptp_rel(dx, dy, dz, da, db, dc)
        if slot is None:
            return False
        return self._wait_slot_done(slot, timeout=timeout)

    def move_lin_rel(self, dx: float = 0, dy: float = 0, dz: float = 0, da: float = 0, db: float = 0, dc: float = 0, timeout: float = 60.0) -> bool:
        """LIN 상대 이동 (완료까지 블로킹)"""
        slot = self.add_move_lin_rel(dx, dy, dz, da, db, dc)
        if slot is None:
            return False
        return self._wait_slot_done(slot, timeout=timeout)

    # ---- 비상정지 / 속도 제어 ----

    def emergency_stop(self) -> bool:
        """소프트 비상정지 (robo_scram=TRUE → 완전 중단, 현재 모션 취소)"""
        logger.warning("비상정지 트리거 (robo_scram)")
        return self.client.write_variable("robo_scram", "TRUE")

    def emergency_stop_release(self) -> bool:
        """비상정지 해제 (robo_scram=FALSE로 인터럽트 다시 활성)"""
        logger.info("비상정지 해제")
        return self.client.write_variable("robo_scram", "FALSE")

    def safety_pause(self) -> bool:
        """안전 일시정지 (robo_safety_over=TRUE → brake f, 재개 가능)"""
        logger.warning("안전 일시정지 트리거 (robo_safety_over)")
        return self.client.write_variable("robo_safety_over", "TRUE")

    def safety_resume(self) -> bool:
        """안전 일시정지 해제 → RESUME으로 원래 모션 이어서 진행"""
        logger.info("안전 일시정지 해제 (재개)")
        return self.client.write_variable("robo_safety_over", "FALSE")

    def set_speed(self, vel_pct: int, acc_pct: Optional[int] = None) -> bool:
        """
        로봇 속도 설정 (1~100%)

        - $OV_PRO (전역 속도 오버라이드): 즉시 모든 모션에 적용 ⭐ 핵심
        - robo_vel_speed/acc_speed[6]: PTP 축속도 (다음 모션부터)
        """
        vel_pct = max(1, min(100, vel_pct))
        if acc_pct is None:
            acc_pct = vel_pct
        acc_pct = max(1, min(100, acc_pct))

        # 1. 전역 속도 오버라이드 (PTP / LIN 모두에 즉시 적용)
        ok_ov = self.client.write_variable("$OV_PRO", str(vel_pct))

        # 2. PTP 축별 속도/가속도 (다음 모션부터)
        for i in range(1, 7):
            self.client.write_variable(f"robo_vel_speed[{i}]", str(vel_pct))
            self.client.write_variable(f"robo_acc_speed[{i}]", str(acc_pct))
        self.client.write_variable("robo_speed_change", "TRUE")

        if ok_ov:
            logger.info(f"속도 설정: $OV_PRO={vel_pct}%, 축속도={vel_pct}%")
        return ok_ov

    def clear_queue(self) -> bool:
        """큐의 모든 슬롯을 0(대기)으로 리셋"""
        success = True
        for i in range(1, self.QUEUE_SIZE + 1):
            if not self.client.write_variable(f"robo_motion_type[{i}]", "0"):
                success = False
        return success

    def read_variable(self, var_name: str) -> Optional[str]:
        """임의의 KRL 변수 읽기"""
        return self.client.read_variable(var_name)

    def write_variable(self, var_name: str, value: str) -> bool:
        """임의의 KRL 변수 쓰기"""
        return self.client.write_variable(var_name, value)


# 사용 예시 / 진단 도구
if __name__ == "__main__":
    ROBOT_IP = "192.168.20.77"

    # Tool 1, Base 0 (NULLFRAME = World) 사용
    robot = KUKARobot(ROBOT_IP, tool_num=1, base_num=0)

    if not robot.connect():
        print("C3Bridge 연결 실패 - 로봇/C3Bridge 서버 확인")
        exit(1)

    print("\n" + "=" * 50)
    print("  KUKA 로봇 진단")
    print("=" * 50)

    # ── 위치 ──
    tcp = robot.get_tcp_position()
    if tcp:
        print(f"\n[TCP 위치]")
        for axis in ["x", "y", "z", "a", "b", "c"]:
            print(f"  {axis.upper()}: {tcp.get(axis, 0):>8.2f}")

    axis_pos = robot.get_axis_position()
    if axis_pos:
        print(f"\n[축 각도]")
        for k in ["a1", "a2", "a3", "a4", "a5", "a6"]:
            print(f"  {k.upper()}: {axis_pos.get(k, 0):>8.2f}")

    # ── 시스템 상태 ──
    print(f"\n[시스템]")
    print(f"  모드          : {robot.read_variable('$MODE_OP')}")
    print(f"  속도 오버라이드: {robot.read_variable('$OV_PRO')}%")
    print(f"  프로그램 상태 : {robot.read_variable('$PRO_STATE')}")
    print(f"  드라이브 ON   : {robot.read_variable('$DRIVES_ON')}")
    print(f"  USER_SAF      : {robot.read_variable('$USER_SAF')}")

    # ── ext_move.src 진단 ──
    print(f"\n[ext_move.src 변수]")
    idx = robot.read_variable("robo_motion_index")
    print(f"  robo_motion_index : {idx}")
    print(f"  robo_scram        : {robot.read_variable('robo_scram')}")
    print(f"  robo_safety_over  : {robot.read_variable('robo_safety_over')}")
    print(f"  robo_speed_change : {robot.read_variable('robo_speed_change')}")
    print(f"  robo_vel_speed[1] : {robot.read_variable('robo_vel_speed[1]')}")
    print(f"  robo_acc_speed[1] : {robot.read_variable('robo_acc_speed[1]')}")

    # 큐 상태
    print(f"\n[모션 큐 상태] (0=대기, 1~4=처리 중)")
    for i in range(1, 21):
        t = robot.read_variable(f"robo_motion_type[{i}]")
        if t and t.strip() != "0":
            mode = robot.read_variable(f"robo_motion_mode[{i}]")
            print(f"  [{i:2d}] type={t}, mode={'E6POS' if mode == 'TRUE' else 'AXIS'}")
    print(f"  (위에 표시 안 된 슬롯은 모두 비어있음)")

    # ── 이동 테스트는 주석으로 유지 (안전) ──
    # robot.set_speed(10)
    # robot.move_lin_rel(dz=5, timeout=30)

    robot.disconnect()
    print("\n" + "=" * 50)
