# Tree Harness

> 项目级认知中间件，为 coding agent 提供跨 episode 的长期知识管理。

Tree Harness 把 agent 对一个项目的认知建模成一棵树：外层（形成层）不断从 trajectory 蒸馏出新的知识细胞（cell），经过验证的知识逐渐木质化（从边材硬化为心材），过时知识则被腐朽检测识别并隔离。由此解决 coding agent 长期使用中的两个痛点：**上下文污染**（对话越长效果越差）与**跨 session 失忆**（新对话什么都不记得）。

系统只存储蒸馏后的 cell（`Context, Decision, Rationale` 三元组），不存储原始 trajectory / 对话 / tool call。

---

## 环境要求

| 项 | 要求 |
|----|------|
| Python | ≥ 3.9（推荐 3.10+，见下方「常见问题」） |
| 操作系统 | macOS / Linux |
| 网络 | 安装依赖与首次加载嵌入模型需联网 |

**核心依赖**：`kuzu`（图数据库）、`sqlite-vec`（SQLite 向量检索）、`sentence-transformers`（嵌入模型）、`openai`（LLM 调用，OpenAI 兼容接口）、`python-dotenv`（环境变量加载）。

---

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/AmazeZhang/tree-harness.git
cd tree-harness
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. 安装依赖

以可编辑模式安装本包及其开发依赖（注意 `zsh` 下需用引号包裹 `".[dev]"`）：

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

这会安装运行依赖（`kuzu`、`sqlite-vec`、`sentence-transformers` 等）以及开发依赖（`pytest`、`pytest-cov`）。

### 4. 配置 LLM 环境变量

复制示例文件并填入你自己的 API 配置：

```bash
cp .env.example .env
```

编辑 `.env`，填入任一 LLM provider 的配置（二选一即可，代码默认使用 Qwen）：

```bash
# Qwen（默认）
QWEN_API_KEY=sk-xxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# 或 DeepSeek
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 项目通过 `RealLLMClient.from_env(provider)` 读取环境变量，命名规则为 `{PROVIDER}_API_KEY` / `{PROVIDER}_BASE_URL` / `{PROVIDER}_MODEL`。切换 provider 需在调用处修改 `from_env(...)` 的参数。

### 5. 验证安装

跑一遍测试套件确认环境就绪（测试使用确定性桩，**不需要** API key 和模型下载）：

```bash
pytest
```

看到全部通过即代表环境配置成功。

---

## 环境变量说明

| 变量 | 说明 | 示例 |
|------|------|------|
| `QWEN_API_KEY` | Qwen API 密钥 | `sk-xxxx` |
| `QWEN_BASE_URL` | Qwen OpenAI 兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `QWEN_MODEL` | Qwen 模型名 | `qwen-plus` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxxx` |
| `DEEPSEEK_BASE_URL` | DeepSeek OpenAI 兼容端点 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | `deepseek-chat` |

> `.env` 已被 `.gitignore` 忽略，不会提交；`.env.example` 是公开模板，可安全共享。

---

## 项目结构

```
tree-harness/
├── src/tree_harness/
│   ├── core/            # 核心数据模型与抽象层
│   │   ├── cell_model.py     # Cell 数据模型（知识细胞）
│   │   ├── embedding.py      # Embedder 抽象（确定性桩 + SentenceTransformer）
│   │   ├── llm_client.py     # LLM 调用抽象（temperature=0，全量缓存）
│   │   └── oplog.py          # OpLog 事件溯源（append + replay）
│   ├── store/           # 存储后端
│   │   ├── sqlite_backend.py # SQLite + sqlite-vec 向量检索
│   │   ├── kuzu_backend.py   # KuzuDB 图存储
│   │   └── tree_store.py     # 统一接口，协调双库
│   ├── modules/         # 业务模块
│   │   ├── cambium_engine.py     # 形成层：trajectory → cell 蒸馏管线
│   │   ├── dedup.py              # 去重（embedding + LLM 仲裁）
│   │   ├── connector.py          # 语义接线（ray 边）
│   │   ├── energy_system.py      # 能量/成熟度更新
│   │   ├── ring_promotion.py     # 年轮层升降（带滞回）
│   │   ├── decay_sentinel.py     # 腐朽检测漏斗
│   │   ├── lignification.py      # 木质化调度（promote/merge/split）
│   │   ├── context_injector.py   # 树 → agent 上下文注入
│   │   ├── verifiers.py          # 验证器（test runner / grep / AST）
│   │   └── outer_harness.py      # 系统对外唯一入口（三个 hook）
│   └── adapters/
│       └── trajectory_adapter.py # SWE-agent trajectory → 标准格式
├── tests/              # 测试套件（pytest）
├── docs/               # 设计文档与各模块 spec
│   ├── DESIGN.md
│   ├── DEVELOPMENT_SPEC.md
│   └── specs/               # 各模块规格文档
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 核心概念

- **Cell（细胞）**：系统的基本认知单位，即 `(Context, Decision, Rationale)` 三元组。细胞不可修改，只能弃用和新生，从而保留完整演变史。
- **Ring（年轮层）**：按知识成熟度分层（L0 外层活跃 → L4 内层稳固），不同层有不同的衰减半衰期。
- **木质化**：成熟度达阈值的 cell 从边材硬化为心材（promote / merge / split）。
- **腐朽检测**：低能量 cell 经漏斗式验证后隔离，避免过时知识污染上下文。
- **OpLog**：所有写操作以事件追加记录，可从 op log 重建任意时刻状态。

详见 [`docs/DESIGN.md`](docs/DESIGN.md)。

---

## 运行测试

```bash
# 全量测试
pytest

# 带覆盖率
pytest --cov=tree_harness

# 仅跑某一模块
pytest tests/test_cambium_engine.py
```

测试配置见 `pyproject.toml`（`testpaths = ["tests"]`，`pythonpath = ["src"]`）。

---

## 常见问题

**Q: `sqlite-vec` 加载报错 / `sqlite3.OperationalError`？**
`sqlite-vec` 是 SQLite 的 C 扩展，对 Python 自带的 SQLite 静态库可能加载失败（常见于 macOS 系统 Python）。建议改用 **Homebrew** 或 **pyenv** 安装的 Python 3.10+：

```bash
brew install python@3.12
# 或
pyenv install 3.12
```

然后基于该解释器重建虚拟环境。

**Q: 首次运行卡在模型下载？**
真实嵌入使用 `BAAI/bge-base-zh-v1.5`（约 400MB），首次实例化 `SentenceTransformerEmbedder` 时会从 HuggingFace 自动下载。测试套件使用 `DeterministicEmbedder`（纯哈希向量），无需下载模型、无需网络。

**Q: 不配置 LLM 能用吗？**
可以。开发/测试场景可使用 `DeterministicLLMClient`（确定性桩），无需 API key。只有调用真实 LLM 蒸馏（`RealLLMClient`）时才需要配置 `.env`。

---

## 文档

- [设计文档](docs/DESIGN.md) — 核心比喻、七条公理、本体定义
- [开发规格](docs/DEVELOPMENT_SPEC.md) — 规格驱动开发的阶段划分
- [模块规格](docs/specs/) — 各模块的接口与行为契约
- [相关工作调研](docs/related-work-survey.md)
