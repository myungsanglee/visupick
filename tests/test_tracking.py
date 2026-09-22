"""tracking.py 테스트 — 합성 시퀀스로 ID 유지·교차·가림·재등장을 확인한다.

추적의 가치는 "같은 물체에 같은 번호가 계속 붙는가" 하나로 요약되므로, 테스트도
프레임 시퀀스를 만들어 **track_id 가 바뀌지 않는지**를 본다. 검출기는 쓰지 않는다
(모듈이 검출기 비종속이라는 것 자체가 설계 목표).
"""

import numpy as np
import pytest

from tracking import BotSortTracker, ByteTrackTracker, SortTracker, Track, create_tracker, giou_matrix, iou_matrix


def det(x, y, w=40, h=30, conf=0.9, cls=0, name="case"):
    return {"bbox": [float(x), float(y), float(x + w), float(y + h)], "confidence": conf, "class_id": cls, "class_name": name}


def run(tracker, frames):
    """프레임마다 update 를 돌려 [{track_id: bbox}, ...] 를 반환."""
    out = []
    for dets in frames:
        res = tracker.update(dets)
        out.append({r["track_id"]: r["bbox"] for r in res})
    return out


@pytest.fixture(autouse=True)
def _reset_ids():
    Track.reset_ids()


class TestGeometry:
    def test_iou_basics(self):
        a = np.array([[0, 0, 10, 10]])
        assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
        assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0
        assert iou_matrix(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)

    def test_giou_distinguishes_far_from_near(self):
        """IoU 는 안 겹치면 전부 0 이지만 GIoU 는 거리를 구분해야 한다."""
        a = np.array([[0, 0, 10, 10]])
        near = giou_matrix(a, np.array([[12, 0, 22, 10]]))[0, 0]
        far = giou_matrix(a, np.array([[100, 0, 110, 10]]))[0, 0]
        assert iou_matrix(a, np.array([[12, 0, 22, 10]]))[0, 0] == iou_matrix(a, np.array([[100, 0, 110, 10]]))[0, 0] == 0.0
        assert near > far

    def test_empty_inputs(self):
        assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
        assert giou_matrix(np.array([[0, 0, 1, 1]]), np.zeros((0, 4))).shape == (1, 0)


@pytest.mark.parametrize("make", [SortTracker, ByteTrackTracker, BotSortTracker])
class TestIdentityAcrossFrames:
    def test_single_object_keeps_id(self, make):
        t = make(min_hits=2)
        frames = [[det(10 + 5 * i, 20)] for i in range(10)]
        res = run(t, frames)
        ids = [list(r)[0] for r in res if r]
        assert ids, "아무 트랙도 확정되지 않음"
        assert len(set(ids)) == 1, f"ID 가 바뀜: {ids}"

    def test_two_objects_get_distinct_ids(self, make):
        t = make(min_hits=2)
        frames = [[det(10 + 4 * i, 20), det(300 - 4 * i, 200)] for i in range(8)]
        res = run(t, frames)
        assert len(res[-1]) == 2
        assert len(set(res[-1])) == 2

    def test_crossing_objects_do_not_swap(self, make):
        """서로 지나쳐 가도 ID 가 뒤바뀌면 안 된다 (헝가리안 할당의 존재 이유)."""
        t = make(min_hits=2)
        frames = []
        for i in range(12):
            frames.append([det(20 + 12 * i, 100), det(152 - 12 * i, 108)])
        res = run(t, frames)
        first, last = res[3], res[-1]
        assert len(first) == len(last) == 2
        # 왼→오 로 가던 트랙이 끝에도 더 오른쪽에 있어야 한다
        moving_right = min(first, key=lambda k: first[k][0])
        assert last[moving_right][0] > first[moving_right][0]

    def test_reappearing_object_keeps_id(self, make):
        """몇 프레임 가려졌다 다시 나타나도 max_age 안이면 같은 ID 여야 한다."""
        t = make(min_hits=2, max_age=10)
        frames = [[det(10 + 5 * i, 20)] for i in range(5)]
        frames += [[] for _ in range(4)]  # 가려짐
        frames += [[det(10 + 5 * i, 20)] for i in range(9, 14)]
        res = run(t, frames)
        before = list(res[4])[0]
        after = list(res[-1])[0]
        assert before == after, f"재등장 후 ID 가 바뀜: {before} → {after}"

    def test_lost_beyond_max_age_gets_new_id(self, make):
        t = make(min_hits=2, max_age=3)
        frames = [[det(10, 20)] for _ in range(4)]
        frames += [[] for _ in range(8)]  # max_age 초과
        frames += [[det(10, 20)] for _ in range(4)]
        res = run(t, frames)
        assert list(res[3])[0] != list(res[-1])[0]


class TestConfirmation:
    def test_min_hits_delays_id(self):
        """한 번 깜빡인 오검출에 ID 를 주지 않는다."""
        t = ByteTrackTracker(min_hits=3)
        assert t.update([det(10, 10)]) == []  # 1회차: 아직 미확정
        assert t.update([det(11, 10)]) == []  # 2회차
        assert len(t.update([det(12, 10)])) == 1  # 3회차에 확정

    def test_min_hits_one_confirms_immediately(self):
        t = ByteTrackTracker(min_hits=1)
        assert len(t.update([det(10, 10)])) == 1


class TestByteTrackTwoStage:
    def test_low_conf_keeps_track_alive(self):
        """저신뢰 검출로도 기존 트랙은 이어진다 (2단계 연관)."""
        t = ByteTrackTracker(min_hits=2, high_thresh=0.5, low_thresh=0.1)
        t.update([det(10, 10, conf=0.9)])
        first = t.update([det(14, 10, conf=0.9)])[0]["track_id"]
        res = t.update([det(18, 10, conf=0.3)])  # 저신뢰
        assert res and res[0]["track_id"] == first

    def test_low_conf_alone_never_creates_track(self):
        """저신뢰 검출만으로는 새 트랙이 생기지 않는다 (오검출 승격 방지)."""
        t = ByteTrackTracker(min_hits=1, high_thresh=0.5, low_thresh=0.1)
        for _ in range(5):
            assert t.update([det(10, 10, conf=0.3)]) == []

    def test_below_low_thresh_ignored(self):
        t = ByteTrackTracker(min_hits=1, low_thresh=0.2)
        assert t.update([det(10, 10, conf=0.05)]) == []


class TestClassAware:
    def test_different_classes_do_not_match(self):
        """쉼표 다중 프롬프트로 여러 종류를 볼 때 종류가 뒤바뀌면 안 된다."""
        t = ByteTrackTracker(min_hits=1, class_aware=True)
        a = t.update([det(10, 10, cls=0, name="rectangle")])[0]["track_id"]
        res = t.update([det(11, 10, cls=1, name="circle")])
        assert not res or res[0]["track_id"] != a

    def test_same_class_matches(self):
        t = ByteTrackTracker(min_hits=1, class_aware=True)
        a = t.update([det(10, 10, cls=1)])[0]["track_id"]
        assert t.update([det(12, 10, cls=1)])[0]["track_id"] == a


class TestContract:
    def test_input_not_mutated_and_fields_preserved(self):
        """입력 dict 를 훼손하지 않고, mask 등 부가 필드는 그대로 넘어가야 한다."""
        t = ByteTrackTracker(min_hits=1)
        mask = np.zeros((20, 20), bool)
        mask[2:8, 2:8] = True
        d = det(10, 10)
        d["mask"] = mask
        out = t.update([d])[0]
        assert "track_id" not in d, "입력이 수정됨"
        assert out["mask"] is mask and out["class_name"] == "case"
        assert out["bbox"] == d["bbox"]

    def test_empty_update_is_safe(self):
        t = ByteTrackTracker()
        assert t.update([]) == []
        assert t.update(None) == []

    def test_reset_clears_tracks(self):
        t = ByteTrackTracker(min_hits=1)
        t.update([det(10, 10)])
        assert t.tracks
        t.reset()
        assert t.tracks == []

    def test_output_order_follows_input(self):
        t = ByteTrackTracker(min_hits=1)
        dets = [det(200, 10), det(10, 10)]
        out = t.update(dets)
        assert [o["bbox"] for o in out] == [d["bbox"] for d in dets]


class TestFactory:
    @pytest.mark.parametrize("name", ["sort", "SORT", "botsort", "BoT-SORT", "bytetrack", "byte_track"])
    def test_create(self, name):
        assert create_tracker(name) is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_tracker("deepsort")

    def test_default_is_botsort_without_cmc(self):
        t = create_tracker()
        assert isinstance(t, BotSortTracker)
        assert t.use_cmc is False, "고정 카메라에서는 CMC 가 기본 꺼짐이어야 한다"


class TestSortIsSingleStage:
    """SORT 는 1단계 연관만 한다 — ByteTrack 과의 차이가 실제로 드러나야 한다."""

    def test_low_conf_does_not_extend_track(self):
        """저신뢰 검출로는 트랙을 잇지 않는다 (ByteTrack 은 잇는다)."""
        t = SortTracker(min_hits=2, high_thresh=0.5)
        t.update([det(10, 10, conf=0.9)])
        t.update([det(14, 10, conf=0.9)])
        assert t.update([det(18, 10, conf=0.3)]) == []

    def test_bytetrack_does_extend_track(self):
        t = ByteTrackTracker(min_hits=2, high_thresh=0.5, low_thresh=0.1)
        t.update([det(10, 10, conf=0.9)])
        t.update([det(14, 10, conf=0.9)])
        assert t.update([det(18, 10, conf=0.3)]), "ByteTrack 은 저신뢰로도 이어야 한다"

    def test_second_stage_flag(self):
        assert SortTracker().second_stage is False
        assert ByteTrackTracker().second_stage is True
        assert BotSortTracker().second_stage is True


@pytest.mark.parametrize("make", [SortTracker, ByteTrackTracker, BotSortTracker])
class TestPruneDuringMatch:
    """만료 트랙 정리가 같은 프레임의 매칭 결과를 망가뜨리지 않는지 (회귀 테스트).

    실제로 났던 버그: 매칭 결과를 self.tracks 의 **인덱스**로 들고 있다가, 만료 트랙을
    걷어내 목록이 짧아진 뒤에 그 인덱스로 되찾았다. 앞쪽 트랙이 사라지면 뒤쪽 인덱스가
    한 칸씩 밀려 엉뚱한 트랙을 가리키고(= ID 뒤바뀜), 맨 뒤였다면 IndexError 로 터진다.
    두 증상 모두 "앞 트랙이 만료되는 그 프레임에 뒤 트랙이 매칭된다"는 한 조건에서 나온다.
    """

    def test_expiring_first_track_does_not_break_later_ones(self, make):
        max_age = 3
        t = make(min_hits=2, max_age=max_age)

        # 먼저 3개를 확정시킨다 → self.tracks = [A, B, C]
        frames = [[det(10, 20), det(300, 20), det(600, 20)] for _ in range(3)]
        res = run(t, frames)
        a_id, b_id, c_id = (list(r)[0] for r in ({k: v for k, v in res[-1].items() if abs(v[0] - x) < 1} for x in (10, 300, 600)))

        # A 만 사라지고 B·C 는 계속 잡힌다 → A 가 만료되는 프레임에 B·C 가 매칭된다
        tail = [[det(300, 20), det(600, 20)] for _ in range(max_age + 3)]
        res = run(t, tail)

        # 만료 프레임을 실제로 지났는지 확인 (조건을 못 만들면 테스트가 무의미)
        assert all(abs(tr.time_since_update) <= max_age for tr in t.tracks)
        assert len(t.tracks) == 2, f"A 가 정리되지 않음: {len(t.tracks)}"

        last = res[-1]
        assert set(last) == {b_id, c_id}, f"ID 가 뒤바뀌거나 사라짐: {sorted(last)} != {sorted((b_id, c_id))}"
        assert a_id not in last
        assert abs(last[b_id][0] - 300) < 50 and abs(last[c_id][0] - 600) < 50, "ID 가 서로 뒤바뀜"
