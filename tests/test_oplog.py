"""OpLog 测试 —— 对应 docs/specs/oplog.md 测试用例。"""
import pytest

from tree_harness.core.oplog import OpLog, OpLogEntry, OpEnum, OP_TO_OPERATOR, ReplayTarget


@pytest.fixture
def oplog():
    log = OpLog(":memory:")
    yield log
    log.close()


# 测试用例 1: append 3 条 → get_entries 返回 3 条, seq 为 1,2,3
def test_append_and_get_entries(oplog):
    s1 = oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1", "ring": "L0"}, "ep1")
    s2 = oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")
    s3 = oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1", "old": 0.5, "new": 0.6}, "ep1")
    assert [s1, s2, s3] == [1, 2, 3]
    entries = oplog.get_entries()
    assert len(entries) == 3
    assert [e.seq for e in entries] == [1, 2, 3]
    assert entries[0].op == OpEnum.INSERT_CELL
    assert entries[2].payload["new"] == 0.6
    assert oplog.get_latest_seq() == 3


# 测试用例 2: get_cell_history 只返回包含该 cell_id 的条目
def test_get_cell_history(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1", "ring": "L0"}, "ep1")
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c2", "ring": "L0"}, "ep1")
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1", "old": 0.5, "new": 0.6}, "ep2")
    oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1"}, "ep2")
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c2", "to_id": "c1"}, "ep2")  # c1 作为 to_id

    history = oplog.get_cell_history("c1")
    # c1 出现在 cell_id / to_id,共 4 条 (c2 的 INSERT_CELL 不含 c1)
    assert len(history) == 4
    ops = {e.op for e in history}
    assert OpEnum.INSERT_RAY in ops  # to_id=c1


# 测试用例 3: get_episode_ops 只返回该 episode 关联的条目
def test_get_episode_ops(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1"}, "ep-10")
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c2"}, "ep-11")
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1"}, "ep-10")
    oplog.append(OpEnum.PROMOTE, {"cell_id": "c2"}, "ep-10")

    ep10 = oplog.get_episode_ops("ep-10")
    assert len(ep10) == 3
    assert all(e.episode_id == "ep-10" for e in ep10)
    assert oplog.get_episode_ops("ep-99") == []


# 测试用例 4: op_filter 过滤正确 (只返回 INSERT_CELL 类型)
def test_get_entries_with_op_filter(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1"}, "ep1")
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c2"}, "ep1")
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1"}, "ep1")

    cells = oplog.get_entries(op_filter=OpEnum.INSERT_CELL)
    assert len(cells) == 2
    assert all(e.op == OpEnum.INSERT_CELL for e in cells)
    assert [e.seq for e in cells] == [1, 3]


# 测试用例 5: replay 从空库重建 → 逐条调用 tree_store.apply_op
def test_replay_invokes_apply_op_in_order(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1", "ring": "L0"}, "ep1")
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1", "old": 0.5, "new": 0.6}, "ep1")

    class FakeTreeStore:
        def __init__(self):
            self.calls = []

        def apply_op(self, op, payload, episode_id):
            self.calls.append((op, payload, episode_id))

    fake = FakeTreeStore()
    oplog.replay(fake)
    assert len(fake.calls) == 3
    assert fake.calls[0][0] == OpEnum.INSERT_CELL
    assert fake.calls[1][0] == OpEnum.INSERT_RAY
    assert fake.calls[2][0] == OpEnum.UPDATE_ENERGY
    assert fake.calls[2][1]["new"] == 0.6


# replay 支持 from_seq / to_seq 区间
def test_replay_range(oplog):
    for i in range(5):
        oplog.append(OpEnum.INSERT_CELL, {"cell_id": f"c{i}"}, "ep1")

    class Fake:
        def __init__(self):
            self.n = 0

        def apply_op(self, op, payload, episode_id):
            self.n += 1

    fake = Fake()
    oplog.replay(fake, from_seq=2, to_seq=4)
    # seq > 2 且 seq <= 4 → seq 3,4 → 2 条
    assert fake.n == 2


def test_replay_target_protocol():
    """ReplayTarget 是 runtime_checkable Protocol。"""
    class Good:
        def apply_op(self, op, payload, episode_id):
            pass

    class Bad:
        pass

    assert isinstance(Good(), ReplayTarget)
    assert not isinstance(Bad(), ReplayTarget)


# 测试用例 6: operator 字段自动派生 (调用方不传, 由 OP_TO_OPERATOR 填充)
def test_operator_auto_derived(oplog):
    """append(INSERT_RAY) → OpLogEntry.operator == "CONNECT", SQLite operator 列 == "CONNECT"。"""
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")
    entries = oplog.get_entries()
    assert len(entries) == 1
    assert entries[0].operator == "CONNECT"
    assert entries[0].op == OpEnum.INSERT_RAY

    # INSERT_CELL → CRYSTALLIZE
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1", "ring": "L0"}, "ep1")
    assert oplog.get_entries()[-1].operator == "CRYSTALLIZE"

    # UPDATE_ENERGY → DECAY
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1"}, "ep1")
    assert oplog.get_entries()[-1].operator == "DECAY"


# 测试用例 7: 未注册 op → KeyError (I-Op2 运行时兜底)
def test_unknown_op_raises_keyerror(oplog):
    with pytest.raises(KeyError):
        oplog.append("NOT_A_REAL_OP", {"cell_id": "c1"}, "ep1")


# 测试用例 8: count_by_op_type 按 5 算符聚合, 缺失算符 0 填充
def test_count_by_op_type(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1"}, "ep1")       # CRYSTALLIZE
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c2"}, "ep1")       # CRYSTALLIZE
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")  # CONNECT
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c2", "to_id": "c1"}, "ep1")  # CONNECT
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c2", "to_id": "c3"}, "ep1")  # CONNECT
    oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1"}, "ep1")     # DECAY

    counts = oplog.count_by_op_type()
    assert counts == {
        "CRYSTALLIZE": 2,
        "CONNECT": 3,
        "PROMOTE": 0,
        "QUARANTINE": 0,
        "DECAY": 1,
    }


# count_by_op_type 按 episode 过滤
def test_count_by_op_type_episode_filter(oplog):
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1"}, "ep1")
    oplog.append(OpEnum.INSERT_RAY, {"from_id": "c1", "to_id": "c2"}, "ep1")
    oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c3"}, "ep2")

    ep1 = oplog.count_by_op_type(episode_id="ep1")
    assert ep1["CRYSTALLIZE"] == 1
    assert ep1["CONNECT"] == 1
    assert ep1["DECAY"] == 0

    ep2 = oplog.count_by_op_type(episode_id="ep2")
    assert ep2["CRYSTALLIZE"] == 1
    assert ep2["CONNECT"] == 0


# count_promotes_by_reason 统计 PROMOTE/DEMOTE 的 reason
def test_count_promotes_by_reason(oplog):
    oplog.append(OpEnum.PROMOTE, {
        "cell_id": "c1", "from_ring": "L0", "to_ring": "L1", "reason": "normal",
    }, "ep1")
    oplog.append(OpEnum.DEMOTE, {
        "cell_id": "c2", "from_ring": "L2", "to_ring": "L1", "reason": "overflow_demote",
    }, "ep1")
    oplog.append(OpEnum.PROMOTE, {
        "cell_id": "c3", "from_ring": "L1", "to_ring": "L2", "reason": "overflow_force",
    }, "ep1")
    oplog.append(OpEnum.PROMOTE, {
        "cell_id": "c4", "from_ring": "L0", "to_ring": "L1", "reason": "normal",
    }, "ep2")

    all_reasons = oplog.count_promotes_by_reason()
    assert all_reasons == {"normal": 2, "overflow_force": 1, "overflow_demote": 1}

    ep1_only = oplog.count_promotes_by_reason(episode_id="ep1")
    assert ep1_only == {"normal": 1, "overflow_force": 1, "overflow_demote": 1}


# count_by_raw_op 按底层 op_type 统计 (区分 PROMOTE vs MERGE/DEMOTE/SPLIT)
def test_count_by_raw_op(oplog):
    oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1"}, "ep1")
    oplog.append(OpEnum.MERGE, {"source_ids": ["c2", "c3"], "target_id": "c4"}, "ep1")
    oplog.append(OpEnum.MERGE, {"source_ids": ["c5", "c6"], "target_id": "c7"}, "ep1")
    oplog.append(OpEnum.DEMOTE, {"cell_id": "c8", "from_ring": "L2", "to_ring": "L1"}, "ep1")
    oplog.append(OpEnum.PROMOTE, {"cell_id": "c9", "from_ring": "L1", "to_ring": "L2"}, "ep2")

    # 全局统计
    raw = oplog.count_by_raw_op()
    assert raw["PROMOTE"] == 2    # 真实 ring promotion
    assert raw["MERGE"] == 2      # merge 操作
    assert raw["DEMOTE"] == 1      # demote 操作
    assert raw["SPLIT"] == 0
    assert raw["INSERT_CELL"] == 0

    # 按 episode 过滤
    ep1 = oplog.count_by_raw_op(episode_id="ep1")
    assert ep1["PROMOTE"] == 1
    assert ep1["MERGE"] == 2
    assert ep1["DEMOTE"] == 1

    ep2 = oplog.count_by_raw_op(episode_id="ep2")
    assert ep2["PROMOTE"] == 1
    assert ep2["MERGE"] == 0

    # 与 count_by_op_type 对比: PROMOTE 算符聚合了 PROMOTE+MERGE+DEMOTE
    ops = oplog.count_by_op_type()
    assert ops["PROMOTE"] == 5  # 2 PROMOTE + 2 MERGE + 1 DEMOTE


# I-Op2 映射表完整性: 每个枚举成员都有映射
def test_op_to_operator_covers_all_ops():
    """import-time assert 的运行时验证: OP_TO_OPERATOR 覆盖全部 OpEnum。"""
    enum_values = {e.value for e in OpEnum}
    mapped_keys = set(OP_TO_OPERATOR.keys())
    assert enum_values == mapped_keys
