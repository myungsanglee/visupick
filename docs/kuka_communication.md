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

### 1.3 먼저 알아야 할 KUKA 용어 (처음 보는 사람용)

- **KRL (KUKA Robot Language)**: KUKA 컨트롤러에서 도는 로봇 전용 언어. `.src`(코드) + `.dat`(데이터) 파일 쌍으로 구성된다. 본 시스템의 [ext_move.src](../krl/ext_move.src)가 KRL 프로그램이다.
- **TCP (Tool Center Point)**: 공구(그리퍼) 끝점. "로봇 위치"라고 하면 보통 이 점의 위치+자세 `(X,Y,Z,A,B,C)`를 말한다. (네트워크의 TCP 와는 무관한 약어 충돌이니 문맥으로 구분.)
- **PTP (Point-To-Point)**: 각 관절이 목표 각도로 "가장 빨리" 도는 모션. TCP가 그리는 경로는 곡선이라 예측이 어렵지만, 빠르고 관절 한계에 안전하다. **큰 이동/큰 자세 변화**에 적합.
- **LIN (Linear)**: TCP가 공간에서 **직선**을 그리도록 관절을 보간하는 모션. 경로가 예측 가능해서 **물체 근처 정밀 접근**에 적합. 대신 특이점(singularity) 근처에서 관절이 급회전할 수 있다.
- **`*_REL`**: 현재 위치 기준 상대 이동 (PTP_REL, LIN_REL).
- **운전 모드** (`$MODE_OP`):
  - **T1** — 수동 감속. 데드맨 스위치(SmartPad 뒷면 안전 스위치)를 잡아야만 움직이고 250mm/s 제한. **모든 테스트는 여기서 시작.**
  - **T2** — 수동 고속 (데드맨 필요, 속도 제한 없음)
  - **AUT** — 자동. 사람이 잡지 않아도 프로그램이 연속 실행. 본 시스템의 실사용 모드 (그래서 소프트웨어 50% 속도 상한을 따로 걸어둠 — 6장)
  - **EXT (AUT_EXT)** — 외부(PLC) 제어 자동
- **`$` 변수**: `$OV_PRO`, `$POS_ACT`, `$OUT[7]` 처럼 `$`로 시작하면 KUKA **시스템 변수**. 그 외(`robo_*`)는 `$config.dat`에 선언한 **사용자 전역 변수**다.
- **SmartPad**: 로봇 티치펜던트(조작 패널). 프로그램 선택/실행, 변수·입출력 표시/수정을 여기서 한다.

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
| `robo_vac_on` | BOOL | 진공 그리퍼 목표 상태 — KRL이 `$OUT[7]`(VAC_ON)에 적용 (3.6절) |
| `robo_vac_blow` | BOOL | 블로우 목표 상태 — KRL이 `$OUT[8]`(VAC_Blow)에 적용 |
| `robo_vac_change` | BOOL | 진공 적용 트리거 (3.6절) |

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

   WAIT SEC 0   ; ★ 물리적 모션 완료까지 대기 (아래 설명)

   robo_motion_type[robo_motion_index] = 0   ; 완료 표시 (= 물리적 도착)
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

**`WAIT SEC 0` 가 왜 필요한가 (advance run 함정):** KRL 인터프리터는 `$ADVANCE = 3` 설정으로 로봇의 실제 움직임보다 **최대 3개 모션 앞서** 코드를 실행한다(선행 실행, advance run). 이게 없으면 `PTP ...` 를 모션 파이프라인에 등록만 하고 곧바로 `robo_motion_type = 0` 줄을 실행해서, **로봇이 아직 이동 중인데 슬롯이 "완료"로 표시**된다. 외부 PC 는 이 0 을 보고 픽 사이클의 진공 ON/OFF 타이밍을 잡기 때문에, 실제로 "이동 중에 진공이 꺼져 물건을 떨어뜨리는" 버그가 났었다. `WAIT SEC 0` 은 선행 실행을 그 자리에 멈추는 표준 KRL 기법 — 메인 실행(물리 모션)이 따라올 때까지 기다리므로, 이후의 `= 0` 은 **물리적 도착 완료**를 의미하게 된다.

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

### 3.6 진공 그리퍼 제어 (robo_vac 계약)

빈 픽킹용 SMC 진공 그리퍼는 디지털 출력 `$OUT[7]`(VAC_ON, 흡착 = 잡기) / `$OUT[8]`(VAC_Blow, 놓기 보조)로 구동된다. 그런데 실측 결과 **C3Bridge 변수 인터페이스는 `$OUT` 쓰기를 거부한다** — 읽기와 커스텀 변수/`$OV_PRO` 쓰기는 되는데, 디지털 출력 쓰기만 거부된다. 반면 **KRL 프로그램 안에서는 `$OUT[7]=TRUE`가 아무 제약 없이 된다.**

그래서 모션 큐와 같은 철학("Python은 변수만 쓰고, 실행은 KRL이")으로 계약을 확장했다. `robo_scram`과 동일한 **플래그 트리거 인터럽트** 패턴이다:

```krl
; ext_move.src
INTERRUPT DECL 83 WHEN robo_vac_change DO robo_vac_DEF()

DEF robo_vac_DEF()
   INTERRUPT OFF 83
   $OUT[7] = robo_vac_on      ; 목표 상태 적용
   $OUT[8] = robo_vac_blow
   robo_vac_change = FALSE    ; 트리거 소비
   INTERRUPT ON 83
END
```

Python 측은 [kuka_robot.py](../kuka_robot.py)의 `set_vacuum()` / `vacuum_blow()` / `vacuum_release()`:

```
1. robo_vac_on = "TRUE"      (목표 상태 쓰기 — 커스텀 변수라 가능)
2. robo_vac_change = "TRUE"  (트리거 → KRL 인터럽트 발화)
3. $OUT[7] 읽기로 실제 적용 확인 ($OUT "읽기"는 되므로 진짜 검증, 1회 재시도 포함)
```

주의할 점:

- 인터럽트는 **ext_move가 실행 중(`$PRO_STATE1 = #P_ACTIVE`)일 때만** 발화한다. T1에서 Start를 놓으면 프로그램이 멈춰 진공 명령도 대기 상태가 된다 (AUT에선 상시 동작).
- 프로그램 (재)시작 시 `robo_vac_on = $OUT[7]`로 초기화한다 — **물체를 잡은 채 재시작해도 떨어뜨리지 않기 위해서다.**
- KRL 인터럽트 조건은 **단순 전역 BOOL이어야 한다.** `WHEN (robo_vac_on <> $OUT[7])` 같은 변수 비교식은 컴파일 에러(SmartPad 파일명 옆 X 표시)가 났다 — 그래서 트리거 변수를 따로 둔다.
- 핸들러에 BRAKE가 없어 모션과 무관 — **이동 중에도 진공 전환 가능** (접근하면서 미리 흡착 시작 같은 활용).

UI: 빈 픽킹 / CAD 매칭 / 표면 추적 탭에 공통으로 "진공 ON / 진공 OFF / 블로우" 버튼이 있다 ([robot_control_mixin.py](../robot_control_mixin.py) `_build_vacuum_row` — 세 탭이 같은 Mixin 을 상속하므로 한 곳에서 관리).

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

KRL이 슬롯의 모션을 끝내면 `motion_type[slot]`을 0으로 리셋하므로, 그 값을 100ms마다 폴링한다. 3.2절의 `WAIT SEC 0` 덕분에 이 0 은 "명령 접수"가 아니라 **물리적 도착 완료**를 뜻한다 — 픽 사이클 실행기([robot_control_mixin.py](../robot_control_mixin.py) `_cycle_tick`)가 진공/블로우 스텝 전에 pending 슬롯이 모두 0 이 되기를 기다리는 것도 이 보장에 기반한다.

### 4.4 비동기(`add_*`) vs 블로킹(`move_*`)

| 메서드 | 동작 |
|---|---|
| [`add_move_lin(...)`](../kuka_robot.py#L489) | 큐에 추가하고 슬롯 번호 반환 (반환 후 바로 다음 작업 가능) |
| [`move_lin(...)`](../kuka_robot.py#L530) | 큐에 추가 + 그 슬롯이 완료될 때까지 블로킹 |

빈 픽킹 탭은 `add_*`를 써서 Approach → Target → Retract 3개를 한 번에 큐에 쌓아 KRL이 연속 실행하게 한다 ([bin_picking_tab.py:1424](../bin_picking_tab.py#L1424)).

---

## 5. 속도 제어

[kuka_robot.py](../kuka_robot.py) `set_speed()`는 **단일 손잡이** 설계다 — 사용자 속도 %는 `$OV_PRO` 하나에만 적용한다:

```python
# 1. 전역 속도 오버라이드 (PTP/LIN 모두에 즉시 적용) — 유일한 속도 손잡이
self.client.write_variable("$OV_PRO", str(vel_pct))

# 2. PTP 축별 속도/가속도는 "고정 기준값" (사용자 %를 곱하지 않음!)
for i in range(1, 7):
    self.client.write_variable(f"robo_vel_speed[{i}]", str(self.PTP_AXIS_VEL_BASE))  # 100
    self.client.write_variable(f"robo_acc_speed[{i}]", str(acc_base))                # 50
self.client.write_variable("robo_speed_change", "TRUE")
```

관련 변수의 역할:

- `$OV_PRO`: 글로벌 속도 오버라이드 (SmartPad 속도 슬라이더와 같은 변수). **PTP, LIN 모두에 즉시 곱해진다.** 사용자 속도 스핀은 이것만 조절한다.
- `robo_vel_speed[i]` → `$VEL_AXIS[i]`: PTP(관절 보간)의 축속도 상한. LIN 에는 영향 없음. **고정 100%.**
- `$VEL.CP` (BAS(#VEL_CP, 1) = 1 m/s): LIN 의 기준 속도. ext_move 시작 시 설정, 이후 $OV_PRO 로만 스케일.

**왜 사용자 %를 축속도에 곱하면 안 되나 (과거 버그):** 예전 구현은 스핀 값(예: 30%)을 `$OV_PRO` 와 `$VEL_AXIS` 양쪽에 다 썼다. 그러면 PTP 의 실효 속도는 30% × 30% = **9%** 로 제곱 감속되는 반면, LIN 은 1 m/s × 30% = 300 mm/s 로 선형 감속 — "원래 가장 빨라야 할 PTP 가 LIN 보다 훨씬 느린" 증상이 났다 (픽 사이클의 Approach/이송이 PTP 라 특히 두드러졌음). 축속도를 고정 기준값으로 두면 PTP/LIN 이 같은 비율로 스케일되어 정상 관계(PTP ≥ LIN)가 유지된다.

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
