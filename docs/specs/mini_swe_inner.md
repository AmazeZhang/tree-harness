# MiniSWEAgentInner Spec

## 概述

将 mini-swe-agent 的 `DefaultAgent` 包装为 Tree Harness 的 `InnerHarnessProtocol` 实现，使 OuterHarness 能够以 step 级粒度控制 agent，在每步之间注入 Tree context 并观测 trajectory。

## 设计原则

- **适配器模式**：不修改 mini-swe-agent 源码，通过包装实现 Protocol
- **InnerHarness 与 OuterHarness 严格分离**：OuterHarness 不感知 mini-swe-agent 的存在
- **inner_kind 可替换**：`"mini-swe-agent"` ↔ `"subprocess"` 切换时 OuterHarness 不变
- **沙箱独立**：Environment 由 mini-swe-agent 提供，可换 Local/Docker/E2B

## 架构

```
TreeHarnessRunner
  └── OuterHarness (已完成)
        └── MiniSWEAgentInner (新建, adapter)
              ├── DefaultAgent (mini-swe-agent 包)
              │     ├── Model (LitellmModel 或 WrappedRealLLM)
              │     └── Environment (LocalEnv / DockerEnv)
              └── MiniSWEState (新建, 持有 messages 引用)
```

## 文件结构

```
src/tree_harness/adapters/
  __init__.py
  trajectory_adapter.py       (已有)
  mini_swe_inner.py           ← 新建: MiniSWEAgentInner + MiniSWEState
  mini_swe_model.py           ← 新建: 包装 RealLLMClient 为 mini-swe Model
```

## 接口映射

### InnerHarnessProtocol ↔ DefaultAgent

| OuterHarness Protocol | mini-swe-agent | 适配方式 |
|----------------------|----------------|----------|
| `reset(task)` | `agent.messages = []; add system+user msg` | 提取 run() 前两行 |
| `step(state)` | `agent.step()` | 直接调用 |
| `is_terminal(state)` | `messages[-1].role == "exit"` | 检查最后消息 |
| `capabilities()` | 硬编码 | 返回固定 InnerCapabilities |

### StepState.augment ↔ messages 注入

```python
class MiniSWEState:
    """持有 agent.messages 引用, augment 往前插入 context。"""

    def augment(self, context: ContextBlock) -> "MiniSWEState":
        # 把 pinned + relevant + warnings 拼成一条 user 消息
        injection = self._format_injection(context)
        self.agent.messages.append({
            "role": "user",
            "content": injection,
        })
        return self
```

### StepObservation ↔ mini-swe messages

```python
def step(self, state) -> StepObservation:
    prev_len = len(self.agent.messages)
    self.agent.step()  # query LLM + execute actions

    # 从新追加的消息中提取 observation
    new_msgs = self.agent.messages[prev_len:]
    last_msg = self.agent.messages[-1]

    return StepObservation(
        action={"summary": _extract_action(new_msgs)},
        result={
            "summary": _extract_output(new_msgs),
            "patch": _extract_patch(new_msgs),
            "tests": _extract_tests(new_msgs),
            "outcome": _extract_outcome(last_msg),
            "duration": 0.0,
            "tokens": 0,
        },
        is_terminal=last_msg.get("role") == "exit",
        outcome=_extract_outcome(last_msg),
        raw_output=new_msgs,
    )
```

## Model 对接

mini-swe-agent 的 `Model` 接口需要实现 `query(messages) -> dict`。两个方案：

### 方案 1：直接用 LitellmModel（推荐）

```python
from minisweagent import LitellmModel

model = LitellmModel(model_name="openai/qwen-...")
```

- 优点：零适配代码，支持所有 provider
- 缺点：与现有 RealLLMClient 两套 token 计费

### 方案 2：包装 RealLLMClient

```python
class WrappedRealLLM(Model):
    def __init__(self, llm_client: RealLLMClient):
        self._llm = llm_client

    def query(self, messages: list[dict]) -> dict:
        # 转换 messages → RealLLMClient 格式
        # 调用 llm_client.complete()
        # 转换 response → mini-swe message dict
        ...
```

- 优点：统一 token 计费
- 缺点：需适配 mini-swe 的 action 解析格式

**决策：先用方案 1（LitellmModel），跑通后视需要再转方案 2。**

## Context 注入策略

### 问题

mini-swe-agent 用线性 messages 历史，注入的 context 会永久留在历史中。如果每步都注入，messages 会无限膨胀。

### 解决方案：替换式注入

```python
class MiniSWEState:
    INJECTION_MARKER = "<!--TREE_INJECTION-->"

    def augment(self, context: ContextBlock):
        # 1. 移除上一步的注入消息（如果有）
        self.agent.messages = [
            m for m in self.agent.messages
            if not m.get("content", "").startswith(self.INJECTION_MARKER)
        ]

        # 2. 插入新的注入消息
        injection = self.INJECTION_MARKER + "\n" + self._format(context)
        self.agent.messages.append({
            "role": "user",
            "content": injection,
        })
```

这样每步只有一条注入消息，不会累积。

## 沙箱策略

### 阶段 1：LocalEnvironment + temp dir（开发验证）

```python
from minisweagent.environments.local import LocalEnvironment

env = LocalEnvironment(cwd=str(tempfile.mkdtemp()), timeout=120)
```

- 直接在本机 subprocess 执行
- cwd 指向临时目录，避免污染源码
- 超时 120s 防卡死

### 阶段 2：DockerEnvironment（安全隔离）

```python
from minisweagent.environments.docker import DockerEnvironment

env = DockerEnvironment(image="swebench/sweb.eval.x86_64:...", ...)
```

### 阶段 3：SWE-bench Docker（评测）

使用 SWE-bench 提供的 Docker 镜像，每题一个预装好 repo 的容器。

## 配置

```python
@dataclass
class MiniSWEConfig:
    # Model
    model_name: str = "openai/qwen-2.5-coder-32b-instruct"
    model_base_url: str = ""           # OpenAI 兼容 endpoint

    # Agent
    step_limit: int = 30               # 每 episode 最大步数
    cost_limit: float = 3.0            # 最大 LLM 费用 ($)
    system_template: str = ""          # 留空用 mini-swe 默认
    instance_template: str = ""        # 留空用 mini-swe 默认

    # Environment
    env_kind: str = "local"            # "local" | "docker"
    env_cwd: str = ""                  # 工作目录 (local)
    env_timeout: int = 120             # 命令超时 (秒)
```

## RunnerConfig 扩展

```python
class RunnerConfig:
    inner_kind: Literal["swe-agent", "openhands", "mini-swe-agent", "mock"] = "mock"

    # mini-swe-agent 专用配置
    mini_swe_config: Optional[MiniSWEConfig] = None
```

## _build_inner 扩展

```python
def _build_inner(self, config: RunnerConfig) -> InnerHarnessProtocol:
    if config.inner_kind == "mock":
        return _MockInner()
    elif config.inner_kind == "mini-swe-agent":
        return _build_mini_swe_inner(config.mini_swe_config or MiniSWEConfig())
    ...
```

## 测试策略

### 单元测试（不依赖 mini-swe-agent 包）

```python
class TestMiniSWEState:
    def test_augment_inserts_injection_message(self): ...
    def test_augment_replaces_previous_injection(self): ...
    def test_is_terminal_detects_exit_role(self): ...

class TestMiniSWEAgentInner:
    def test_reset_initializes_messages(self): ...
    def test_step_returns_observation(self): ...
    def test_capabilities_reports_correctly(self): ...
```

### 集成测试（需 mini-swe-agent 包）

```python
def test_end_to_end_one_episode(self):
    """真实跑一个 task, 验证 message 流转 + StepObservation 提取。"""
    ...
```

## 依赖

```
pip install mini-swe-agent    # 提供 DefaultAgent, Model, Environment
pip install litellm            # mini-swe 的 Model 后端
```

## 实现顺序

1. `mini_swe_model.py` — WrappedRealLLM 或直接用 LitellmModel
2. `mini_swe_inner.py` — MiniSWEState + MiniSWEAgentInner
3. `runner.py` — _build_inner 支持 "mini-swe-agent"
4. 单元测试
5. 集成测试（1 个 task 端到端）
