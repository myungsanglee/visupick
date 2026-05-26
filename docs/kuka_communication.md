# KUKA 로봇 통신 (C3Bridge + KRL 큐)

이 문서는 본 프로그램이 KUKA 로봇과 어떻게 통신하는지를 설명한다. 두 층으로 나뉜다:

1. **C3Bridge 프로토콜** — Python에서 로봇 컨트롤러로 명령을 보내고 변수를 읽고 쓰는 저수준 통신.
2. **KRL 모션 큐 (`ext_move.src`)** — 로봇 컨트롤러 안에서 돌아가면서 외부 명령을 받아 실제 이동을 수행하는 KRL 프로그램.

이 두 층이 어떻게 맞물려서 "Python에서 `move_lin(x, y, z)`을 호출하면 로봇이 직선으로 움직이는" 결과가 나오는지를 단계별로 보자.

---

## 1. 왜 이런 구조인가?

### 1.1 KUKA의 일반적인 외부 제어 방식

KUKA에서 외부 PC로 로봇을 제어하는 방법은 보통 다음 중 하나다:

| 방식 | 특징 |
|---|---|
| **KUKA.Ethernet KRL XML** | 공식 옵션, 라이선스 필요, 실시간 통신 |
| **RSI (Robot Sensor Interface)** | 4ms 주기 실시간 보간, 별도 라이선스, 매우 비쌈 |
| **C3Bridge / KukaVarProxy** | 비공식 오픈소스 프로토콜, 변수 읽기/쓰기 + 모션 명령 |
| **OPC UA** | 공식 옵션, 비교적 신식 |

본 프로젝트는 **C3Bridge 방식**을 쓴다. 이유:
- 무료 / 추가 라이선스 불필요.
- 컨트롤러에 KukaVarProxy 호환 서버를 띄워두기만 하면 된다.
- 변수 읽기/쓰기로 거의 모든 KRL 변수에 접근할 수 있다 (`$POS_ACT`, `$OV_PRO`, 사용자 변수 등).
- 단점: 비공식이라 문서가 부족하고, 명령이 ms 단위로 들어가지는 않는다 (수십 ms 지연).

### 1.2 왜 KRL 프로그램 (`ext_move.src`)도 필요한가?

C3Bridge의 "Motion 명령(Type 11)"으로 직접 로봇을 움직일 수도 있다. 그런데 이 방식은:
- 한 번에 한 모션밖에 못 보낸다.
- 비상정지/재개를 깔끔히 처리하기 어렵다.
- 속도/가속도 같은 파라미터를 매번 다시 설정해야 한다.

그래서 **로봇 컨트롤러 안에 작은 서비스 프로그램**(`ext_move.src`)을 띄워놓고, Python은 **변수만 쓰고** KRL이 그 값을 보고 알아서 움직이게 하는 구조를 채택했다. 이게 이전 담당자가 잡아놓은 구조이며, 본 프로그램도 같은 패턴을 그대로 사용한다.

```
[Python (PC)]                    [KRC5 Controller]
                                  
KUKARobot ──── TCP:7000 ────► C3Bridge Server
   │                              │
   │ write_variable(...)          │ (변수 쓰기)
   │                              ▼
   │                        $config.dat의 robo_*[20] 변수들
   │                              │
   │                              │ (KRL이 변수 감시)
   │                              ▼
   │                        ext_move.src (LOOP 중)
   │                              │
   └─────────────────────────────►│ (motion_type ≠ 0이면 PTP/LIN 실행)
                                  ▼
                                실제 모터 구동
```

---

## 2. C3Bridge 프로토콜 (저수준)

구현 위치: [kuka_robot.py:18](../kuka_robot.py#L18) `C3BridgeClient` 클래스.

### 2.1 연결 정보

- **TCP 포트**: 7000
- **인코딩**: 헤더는 빅엔디안 바이너리, 변수명은 ASCII, 위치 문자열은 UTF-16 LE.

### 2.2 메시지 포맷

모든 요청과 응답은 다음과 같이 시작한다:

```
| Tag (2B, BE) | MsgLength (2B, BE) | MsgType (1B) | Payload ... |
```

- `Tag`: 요청-응답 매칭용. 매번 1씩 증가시켜서 보내고 같은 값으로 돌아온다.
- `MsgLength`: 그 뒤 페이로드 길이.
- `MsgType`: 어떤 종류의 메시지인지 식별 (아래 표).

| MsgType | 용도 | 정의 위치 |
|:-:|---|---|
| `0` | KRL 변수 읽기 | [kuka_robot.py:33](../kuka_robot.py#L33) `MSG_READ` |
| `1` | KRL 변수 쓰기 | [kuka_robot.py:34](../kuka_robot.py#L34) `MSG_WRITE` |
| `11` | 모션 명령 | [kuka_robot.py:35](../kuka_robot.py#L35) `MSG_MOTION` |

응답 끝에는 항상 `ErrorCode (2B) + Success (1B)`가 붙는다.

### 2.3 변수 읽기 (Type 0)

[kuka_robot.py:101](../kuka_robot.py#L101) `read_variable()`:

요청:
```
| Tag(2B) | MsgLen(2B) | Type=0 (1B) | NameLen(2B) | Name(ASCII) |
```

응답:
```
| Tag(2B) | MsgLen(2B) | Type=0 (1B) | ValueLen(2B) | Value(ASCII) | ErrCode(2B) | Success(1B) |
```

핵심 코드:
```python
payload = struct.pack(">BH", self.MSG_READ, name_len) + name_bytes
message = struct.pack(">HH", tag, len(payload)) + payload
self.socket.sendall(message)
# 응답에서 value_len을 읽고 그만큼 잘라서 .decode("ascii")
```

예시:
```python
robot.read_variable("$POS_ACT")
# → "{X 1234.5, Y -56.7, Z 800.0, A 90.0, B 0.0, C -180.0, S 6, T 35}"
```

### 2.4 변수 쓰기 (Type 1)

[kuka_robot.py:155](../kuka_robot.py#L155) `write_variable()`:

요청:
```
| Tag(2B) | MsgLen(2B) | Type=1 (1B) | NameLen(2B) | Name | ValueLen(2B) | Value(ASCII) |
```

KRL 값을 문자열로 보낸다. 예를 들어 `BOOL`은 `"TRUE"` / `"FALSE"`, `INT`는 `"50"`, `FRAME`은 `"{X 100, Y 0, Z 200, A 0, B 0, C 180}"`.

```python
robot.write_variable("$OV_PRO", "30")            # 속도 30%
robot.write_variable("robo_scram", "TRUE")       # 비상정지 ON
```

### 2.5 모션 명령 (Type 11)

[kuka_robot.py:203](../kuka_robot.py#L203) `send_motion()`:

```
| Tag(2B) | MsgLen(2B) | Type=11 | MotionType(1B) | PosStrLen(2B) | PosStr(UTF-16 LE) |
```

여기서 위치 문자열만 **UTF-16 LE**로 인코딩된다는 점에 주의. 다른 메시지는 모두 ASCII.

> **본 프로그램은 모션 명령(Type 11)을 직접 쓰지 않는다.** 대신 변수 쓰기(Type 1)로 KRL 큐에 모션을 채워넣고 KRL이 알아서 처리한다 (3장 참고). `send_motion()`은 fallback 또는 직접 제어용으로 남아 있을 뿐.

### 2.6 KRL 위치 문자열 ↔ Python dict 변환

KRL의 FRAME 출력 형식:
```
{X 100.0, Y 200.0, Z 300.0, A 0.0, B 0.0, C 180.0, S 6, T 35}
```

이를 정규식으로 파싱하는 게 [kuka_robot.py:292](../kuka_robot.py#L292) `_parse_krl_position()`:

```python
pattern = r"([A-Za-z]\d*)\s+([-+]?\d*\.?\d+)"
matches = re.findall(pattern, pos_str)
for key, val in matches:
    result[key.lower()] = float(val)
```

거꾸로 dict → KRL 문자열은 [kuka_robot.py:312](../kuka_robot.py#L312) `_format_krl_position()`. 깔끔히 양방향이 된다.

---

## 3. KRL 모션 큐 (`ext_move.src`)

이제 Python이 어떤 변수에 어떻게 써야 로봇이 움직이는지를 보자. 핵심은 [krl/ext_move.src](../krl/ext_move.src)에 있다.

### 3.1 큐 구조

20개 슬롯의 배열을 사용한다. 모든 변수는 `$config.dat`에 미리 선언되어 있다 (이전 담당자가 만들어놓은 구조).

| 변수 | 타입 | 의미 |
|---|---|---|
| `robo_motion_index` | INT | KRL이 지금 처리하려는 슬롯 (1~20, 순환) |
| `robo_motion_type[20]` | INT | 슬롯의 모션 종류 (0=빈 슬롯/완료, 1=PTP, 2=PTP_REL, 3=LIN, 4=LIN_REL) |
| `robo_motion_mode[20]` | BOOL | 좌표 형식 (TRUE=E6POS/카르테시안, FALSE=AXIS/관절) |
| `robo_E6POS[20]` | FRAME | 카르테시안 목표 위치 (X,Y,Z,A,B,C) |
| `robo_E6AXIS[20]` | E6AXIS | 관절 각도 목표 (A1~A6 + 외부축 E1~E6) |
| `robo_scram` | BOOL | 비상정지 트리거 |
| `robo_safety_over` | BOOL | 안전 일시정지 트리거 |
| `robo_speed_change` | BOOL | 속도 변경 요청 플래그 |
| `robo_vel_speed[6]` | INT | 축별 속도 % |
| `robo_acc_speed[6]` | INT | 축별 가속도 % |

### 3.2 KRL 메인 루프

[krl/ext_move.src:82](../krl/ext_move.src#L82):

```krl
DEF robo_move()
   INT i
   CONTINUE
   WAIT FOR (robo_motion_type[robo_motion_index] <> 0)

   ; 속도 변경 요청 처리
   IF robo_speed_change THEN
      FOR i = 1 TO 6
         $VEL_AXIS[i] = robo_vel_speed[i]
         $ACC_AXIS[i] = robo_acc_speed[i]
      ENDFOR
      robo_speed_change = FALSE
   ENDIF

   IF robo_motion_mode[robo_motion_index] THEN
      robo_move_E6POS()    ; 카르테시안 모션
   ELSE
      robo_move_AXIS()     ; 관절 모션
   ENDIF

   robo_motion_type[robo_motion_index] = 0   ; 완료 표시
   robo_motion_index = robo_motion_index + 1
   IF (robo_motion_index == 21) THEN
      robo_motion_index = 1
   ENDIF
END
```

핵심 동작:

1. `WAIT FOR (motion_type[index] <> 0)`: 현재 인덱스 슬롯에 명령이 들어올 때까지 멈춰있음.
2. 명령이 들어오면 mode 플래그를 보고 카르테시안 또는 관절 모션 함수를 호출.
3. 실행 완료되면 그 슬롯의 `motion_type`을 0으로 리셋.
4. 인덱스를 다음 슬롯으로 (20 → 1로 순환).

이 함수는 메인의 `LOOP { robo_move() } ENDLOOP`로 무한 반복된다.

### 3.3 카르테시안 모션 분기

[krl/ext_move.src:127](../krl/ext_move.src#L127):

```krl
DEF robo_move_E6POS()
   CONTINUE
   SWITCH robo_motion_type[robo_motion_index]
      CASE 1   ; PTP - 현재 위치와 다를 때만 (안 그러면 KSS가 에러)
         IF (($POS_ACT.X <> robo_E6POS[index].X) OR ...) THEN
            PTP robo_E6POS[robo_motion_index]
         ENDIF
      CASE 2   ; PTP_REL
         PTP_REL robo_E6POS[robo_motion_index]
      CASE 3   ; LIN
         IF (...같은 안전 체크...) THEN
            LIN robo_E6POS[robo_motion_index]
         ENDIF
      CASE 4   ; LIN_REL
         LIN_REL robo_E6POS[robo_motion_index]
   ENDSWITCH
END
```

PTP/LIN 시 현재 위치와 동일한 목표를 주면 KUKA가 에러를 내므로 차이가 있을 때만 실행한다.

### 3.4 비상정지 인터럽트

[krl/ext_move.src:151](../krl/ext_move.src#L151):

```krl
DEF robo_scram_DEF()
   INTERRUPT OFF 81
   BRAKE F                          ; 즉시 정지
   WAIT FOR (robo_scram == FALSE)   ; 외부에서 해제할 때까지 대기
   INTERRUPT ON 81
   RESUME                           ; 현재 모션을 취소하고 LOOP 복귀
END
```

`BRAKE F`만 있으면 멈췄다가 해제되면 멈춘 위치에서 **그 모션을 이어서** 진행한다. 그래서 `RESUME`을 함께 쓴다 — `RESUME`은 현재 모션을 **완전히 취소**하고 인터럽트가 발생한 함수의 호출자(여기선 메인 LOOP)로 돌아간다. 결과적으로 "비상정지 → 큐 비우기 → 해제" 시퀀스가 깔끔히 동작한다.

`robo_safety_over`는 비슷하지만 `RESUME` 후에도 같은 모션을 다시 시작하지 않는 점이 다르다 (사실 안전 일시정지는 의미상 "재개 가능"이지만, 본 프로그램의 안전 시나리오에 맞춰 같은 패턴으로 구현되어 있다 — [krl/ext_move.src:161](../krl/ext_move.src#L161)).

### 3.5 KRL 프로그램 시작 시 초기화

[krl/ext_move.src:42](../krl/ext_move.src#L42):

```krl
BAS(#BASE, 0)         ; Base 0 = World
BAS(#TOOL, 1)         ; Tool 1 사용
BAS(#VEL_CP, 1)       ; LIN 속도 1 m/s
BAS(#ACC_PTP, 10)     ; PTP 가속 10 m/s²
BAS(#ACC_CP, 100)     ; LIN 가속 100 m/s²

FOR i = 1 TO 6
   robo_vel_speed[i] = 50    ; 시작은 무조건 50%로 초기화
   robo_acc_speed[i] = 50
   $VEL_AXIS[i] = robo_vel_speed[i]
   $ACC_AXIS[i] = robo_acc_speed[i]
ENDFOR

FOR i = 1 TO 20
   robo_motion_mode[i] = FALSE
   robo_motion_type[i] = 0       ; 큐 모두 비움
   ...
ENDFOR

PTP $POS_ACT      ; BCO (Block Coincidence) - 현재 위치 인정
LOOP
   robo_move()
ENDLOOP
```

`PTP $POS_ACT`는 KUKA의 BCO 강제용 더미 모션 — KSS가 "프로그램이 현재 자세에 있다"고 인정하게 만든다. 이걸 안 하면 첫 모션 실행 시 BCO 에러가 난다.

---

## 4. Python 측 큐 인터페이스

`KUKARobot` 클래스가 위 KRL 변수들을 깔끔하게 감춘다.

### 4.1 빈 슬롯 찾기

[kuka_robot.py:382](../kuka_robot.py#L382) `_find_empty_slot()`:

```python
idx_str = self.client.read_variable("robo_motion_index")
cur_idx = int(idx_str.strip()) if idx_str else 1

# cur_idx부터 시작해서 순환 검색
for offset in range(self.QUEUE_SIZE):
    slot = ((cur_idx - 1 + offset) % self.QUEUE_SIZE) + 1
    val = self.client.read_variable(f"robo_motion_type[{slot}]")
    if val is not None and val.strip() == "0":
        return slot
```

**왜 `robo_motion_index`부터 시작하느냐?** KRL은 `motion_index`가 가리키는 슬롯에서 `WAIT FOR`로 대기 중이다. 그래서 그 슬롯(또는 그 직후 슬롯)에 채워야 즉시 처리가 시작된다. 만약 멀리 떨어진 슬롯에 채우면 KRL이 거기까지 다른 빈 슬롯을 다 지나가야(=순환해야) 처리된다.

### 4.2 모션 등록

[kuka_robot.py:409](../kuka_robot.py#L409) `_enqueue_motion()`:

쓰는 순서가 중요하다:

```python
# 1. 좌표 쓰기 (FRAME)
self.client.write_variable(f"robo_E6POS[{slot}]", "{FRAME: X ...}")
# 2. 모드 쓰기 (TRUE = E6POS, FALSE = AXIS)
self.client.write_variable(f"robo_motion_mode[{slot}]", "TRUE")
# 3. 마지막에 motion_type 쓰기 → 0이 아닌 값이 들어가는 순간 KRL의 WAIT FOR가 풀린다
self.client.write_variable(f"robo_motion_type[{slot}]", str(motion_type))
```

motion_type을 마지막에 써야 KRL이 위치/모드가 다 준비된 상태에서 실행을 시작한다. 순서를 바꾸면 motion_type이 먼저 세팅되어 KRL이 미처 채워지지 않은 좌표로 움직이려 할 수 있다.

### 4.3 완료 대기

[kuka_robot.py:454](../kuka_robot.py#L454) `_wait_slot_done()`:

```python
while time.time() - start < timeout:
    val = self.client.read_variable(f"robo_motion_type[{slot}]")
    if val is not None and val.strip() == "0":
        return True
    time.sleep(0.1)
```

KRL이 슬롯의 모션을 끝내면 `motion_type[slot]`을 0으로 리셋하므로, 그 값을 100ms마다 폴링한다.

### 4.4 비동기(`add_*`) vs 블로킹(`move_*`)

| 메서드 | 동작 |
|---|---|
| [`add_move_lin(...)`](../kuka_robot.py#L489) | 큐에 추가하고 슬롯 번호 반환 (반환 후 바로 다음 작업 가능) |
| [`move_lin(...)`](../kuka_robot.py#L530) | 큐에 추가 + 그 슬롯이 완료될 때까지 블로킹 |

빈 픽킹 탭은 `add_*`를 써서 Approach → Target → Retract 3개를 한 번에 큐에 쌓아 KRL이 연속 실행하게 한다 ([bin_picking_tab.py:1424](../bin_picking_tab.py#L1424)).

---

## 5. 속도 제어

[kuka_robot.py:573](../kuka_robot.py#L573) `set_speed()`는 두 가지를 동시에 한다:

```python
# 1. 전역 속도 오버라이드 (PTP/LIN 모두에 즉시 적용)
self.client.write_variable("$OV_PRO", str(vel_pct))

# 2. PTP 축별 속도 (다음 모션부터 - robo_speed_change 플래그로 트리거)
for i in range(1, 7):
    self.client.write_variable(f"robo_vel_speed[{i}]", str(vel_pct))
    self.client.write_variable(f"robo_acc_speed[{i}]", str(acc_pct))
self.client.write_variable("robo_speed_change", "TRUE")
```

**왜 두 개를 다 쓰느냐?**

- `robo_vel_speed[i]` → `$VEL_AXIS[i]`: PTP 같은 관절 보간 모션의 축속도. **LIN 모션에는 영향이 거의 없다.**
- `$OV_PRO`: 글로벌 속도 오버라이드(SmartPAD의 속도 슬라이더와 같은 변수). **PTP, LIN 모두에 즉시 적용된다.**

이전 버전에서는 `robo_vel_speed`만 썼는데, LIN 모션 속도가 안 바뀌는 문제가 있었다 ($VEL_CP를 안 건드렸기 때문). `$OV_PRO`를 추가한 뒤 둘 다 정상 작동.

---

## 6. AUT 모드 안전장치

본 프로그램은 KUKA 모드(T1, T2, AUT, AUT_EXT)를 주기적으로 읽는다 (`_refresh_mode_display()`).

```python
raw = self.main.robot.read_variable("$MODE_OP")
mode = normalize_robot_mode(raw)
```

### 6.1 모드 정규화 + AUT 판정 공통 함수

KUKA `$MODE_OP` 응답은 `'#AUT'`, `' #T1 '`, `'#EXT'` 등 prefix `#` 와 공백/대소문자가 일관되지 않다. 이전엔 두 탭이 각자 substring/exact 매칭을 써서 `#EXT` 같은 케이스에서 한쪽 탭만 AUT 로 인식하는 안전 불일치가 있었다. [kuka_robot.py](../kuka_robot.py) 모듈에 공통 헬퍼를 두어 통일:

```python
def normalize_robot_mode(raw):
    """'#AUT ' → 'AUT' 처럼 정규화 (대문자, # 제거, 공백 제거)."""
    if not raw:
        return "?"
    return raw.strip().upper().replace("#", "")

def is_auto_mode(mode):
    """AUT, AUT_EXT, EX, EXT 모두 자동 모드로 간주."""
    m = (mode or "").upper()
    return "AUT" in m or m.startswith("EX")
```

두 탭의 `_is_aut_mode()` / `_refresh_mode_display()` 가 모두 이 함수를 사용 → AUT 50 % 속도 상한이 어느 탭에서든 일관되게 적용됨.

### 6.2 속도 제한 + Z 한계

```python
def _effective_speed(self, requested):
    if self._is_aut_mode():
        return min(requested, 50)
    return requested
```

`_validate_z()` 는 작업대 표면 밑으로 내려가는 모션을 차단. UI 의 `z_min_spin` 값이 한계.

**개선된 부분**: 시퀀스 큐 `_start_sequence` 도 각 액션의 target Z **+ 계산된 approach 지점 Z** 를 모두 검증한다 (Tool +Z 가 옆/위를 향하면 approach 가 바닥 아래로 갈 수 있음 — 단일 모션 경로와 동일하게 안전).

### 6.3 비상정지 해제 동작

`emergency_stop_release()` 는 `robo_scram=FALSE` 만 쓴다. KRL 의 `robo_scram_DEF` 안에 `RESUME` 이 있어서 현재 모션이 자동 취소되므로 큐의 다음 슬롯이 그대로 재개되지는 않는다. **큐를 비우려면 별도의 "큐 비우기" 버튼**을 사용 — UI 일관성 + 사용자 선택권 보장.

---

## 7. 공통 로봇 제어 Mixin

빈 픽킹 탭과 CAD 매칭 탭이 공유하는 로봇 제어 / 시퀀스 큐 / 안전 메서드 ~19개는 [robot_control_mixin.py:`RobotControlMixin`](../robot_control_mixin.py) 에 추출되어 있다. 두 탭은:

```python
class BinPickingTab(RobotControlMixin, QWidget): ...
class CADMatchingTab(RobotControlMixin, QWidget): ...
```

로 Mixin 을 상속. MRO 가 `RobotControlMixin → QWidget` 순이라 Mixin 메서드가 우선 적용되고, 탭별로 다른 `_refresh_mode_display` 같은 메서드는 각 탭 클래스에 그대로 남아 있어 자연스럽게 해결된다.

이 구조의 이점:
- 안전 관련 수정(예: AUT 감지 통일, e-stop 해제 동작) 이 한 곳에서 끝남
- 두 탭의 동작이 **드리프트하지 않음** (이전엔 따로 수정되다 안전 동작이 어긋난 적 있음)
- 시퀀스 라벨 같은 작은 차이는 클래스 속성으로 파라미터화: `SEQ_OBJECT_NOUN = "객체"` (bin) / `"인스턴스"` (cad)

자세한 사용 패턴은 [docs/bin_picking.md § 10.5](bin_picking.md) 참고.

---

## 8. 전체 흐름 정리

"빈 픽킹 탭에서 객체 클릭 후 이동 버튼" 시나리오의 전체 흐름:

```
[Python] BinPickingTab._execute_move()
    │
    ├─ self.main.robot.set_speed(speed)
    │   ├─► write_variable("$OV_PRO", "30")
    │   └─► write_variable("robo_vel_speed[1..6]", "30")
    │       write_variable("robo_speed_change", "TRUE")
    │
    ├─ self.main.robot.add_move_ptp(approach_x, y, z, ...)  # Approach
    │   ├─ _find_empty_slot() → slot 5 (예시)
    │   ├─► write_variable("robo_E6POS[5]", "{FRAME: X ...}")
    │   ├─► write_variable("robo_motion_mode[5]", "TRUE")
    │   └─► write_variable("robo_motion_type[5]", "1")
    │       └── 이 순간 KRL의 WAIT FOR 해제!
    │
    ├─ self.main.robot.add_move_lin(target_x, ...)         # Target
    │   └─► slot 6에 동일 패턴
    │
    └─ self.main.robot.add_move_lin(approach_x, ...)        # Retract
        └─► slot 7에 동일 패턴

[KRC5] ext_move.src LOOP
    1. WAIT FOR (motion_type[5] ≠ 0)  → 위에서 type 들어옴, 통과
    2. robo_speed_change=TRUE 보고 $VEL_AXIS / $ACC_AXIS 업데이트
    3. mode[5]=TRUE → robo_move_E6POS()
    4. type[5]=1 → PTP robo_E6POS[5]   (실제 모터 구동)
    5. type[5]=0 으로 리셋, index=6
    6. WAIT FOR (motion_type[6] ≠ 0)  → 이미 type 들어와 있음, 즉시 통과
    7. ... 반복
```

Python 호출 한 줄 = KRL 변수 3개 쓰기 = 한 모션 실행. 이게 본 시스템의 통신 패턴이다.

---

## 9. 실용적 디버깅 팁

- **연결 진단**: [kuka_robot.py:616](../kuka_robot.py#L616)의 `if __name__ == "__main__":` 블록을 직접 실행하면 TCP/축/모드/큐 상태를 한 번에 출력해준다.
- **변수 이름 못 찾음 에러**: KRL에서 변수가 `$config.dat`에 선언되지 않은 경우. 또는 인덱스가 범위를 벗어남.
- **이동 명령은 갔는데 안 움직임**: KRL 프로그램이 SmartPAD에서 선택+실행 중인지 확인. AUT 모드가 아니면 데드맨 스위치 + 시작 버튼 필요. `$DRIVES_ON`도 확인.
- **모션 큐가 막힘**: 어떤 슬롯이 0으로 안 돌아갔는지 진단 출력으로 확인. `clear_queue()` 호출하면 모두 0으로 리셋.
- **C3Bridge 응답 지연**: 1회 변수 R/W에 약 10~30ms. 빠른 폴링은 무리. 100ms 정도 간격이 적당.

---

## 10. 참고

- [krl/ext_move.src](../krl/ext_move.src) — KRL 프로그램 본체
- [krl/ext_move.dat](../krl/ext_move.dat) — DEFDAT (변수 외부 참조 선언)
- [kuka_robot.py](../kuka_robot.py) — Python 측 클라이언트
- KukaVarProxy 원본 (참고): https://github.com/ImtsSrl/KUKAVARPROXY
- KUKA System Software (KSS) 8.7 매뉴얼 — `$OV_PRO`, `$VEL_AXIS`, `BAS()`, `RESUME` 등 KRL 명령 레퍼런스
