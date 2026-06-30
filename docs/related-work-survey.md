# 自进化系统调研（Related Work 备料）

> 起点是 datawhalechina/hello-agents 的 Extra10《Agent 自进化》四类闭环综述：
> https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra10-Agent%E8%87%AA%E8%BF%9B%E5%8C%96.md
>
> 本文档把综述里的 10 个方法逐一查清原始出处，并按 tree 项目的"harness / self-evolution / loop / evaluation"四轴做对照，给后续 Related Work 写作存料。

---

## 一、按"被进化的对象"做整体定位

| 编号 | 方法 | 进化对象 | 反馈来源 | 与 tree 的关系 |
|------|------|----------|----------|----------------|
| 1 | Hermes Agent | 记忆 + 偏好 + soul 文件 | 用户纠正 + 反思 | 不对标（memory framing 陷阱） |
| 2 | Agent Zero | 项目内工具/子 agent 配置 | 任务结果 | 不对标 |
| 3 | Darwin Gödel Machine | agent 代码 + 工具集合 | 编程 benchmark 实测 | Related Work 第一段重点对标 |
| 4 | JiuwenClaw | SKILL.md（在线增量） | 规则信号检测器 | Related Work 第一段次重点 |
| 5 | EvoSkill | 整套 agent program（skill + prompt） | 离线 benchmark + Pareto Frontier | Related Work 第一段重点对标 |
| 6 | Ultron | 跨 agent 共享 memory/skill/harness | 轨迹回流 + 命中率衰减 | Related Work 第二段借叙事框架 |
| 7 | OpenSpace | skill 版本谱系（DAG） | 三类触发器 | Related Work 第二段对标 |
| 8 | SkillClaw | 跨设备共享 SKILL.md | API 代理拦截 + PRM | Related Work 第二段次要引用 |
| 9 | OpenClaw-RL | LoRA 权重 | 在线对话 + PRM/Judge | Related Work 第三段 orthogonal axis |
| 10 | Agent Lightning | prompt / 权重等"resource" | spans 总线 | Related Work 第三段 orthogonal axis |

---

## 二、逐项档案

### 1. Hermes Agent
- **出处**：MindStudio 团队博客（非论文），https://www.mindstudio.ai/blog/hermes-agent-five-pillars-memory-skills-soul-crons
- **五支柱**：Memory（episodic/semantic/working 三层 + RAG）、Skills（可组合插件）、Soul（角色/价值/边界文件）、Crons（主动调度）、Self-improvement（反思 + 用户确认）
- **进化机制**：reflection step 周期性回顾，提出 config 更新但需要 user approval
- **与 tree 的差异**：本质是 memory + soul 的版本化，没有结构化算符层，也没有能量动力学；属于工程化 long-term memory 系统
- **写论文时定位**：introduction 一句话带过——"prior work treats it as long-term memory"——别展开

### 2. Agent Zero
- **出处**：开源框架，https://github.com/agent0ai/agent-zero
- **架构**：Project（隔离 workspace/memory/secrets/repos）+ Chat（active session + Agent Profile/Model Preset）+ Skills（按需加载）+ Subordinate Agents（递归委派）
- **"动态有机"含义**：prompts/tools/plugins 全部明文暴露在仓库目录下，可被替换；并非自动学习——是给开发者改的脚手架
- **与 tree 的差异**：进化对象是"开发者手改的配置"，不是闭环自动进化；属于框架层而非方法学
- **写论文时定位**：同 Hermes，introduction 简提即可

### 3. Darwin Gödel Machine (DGM)
- **出处**：Zhang/Hu/Lu/Lange/Clune, arXiv 2505.22954（2025-05；v3 2026-03），代码 https://github.com/jennyzzt/dgm
- **核心算法**：维护一棵 agent archive；每轮基础模型从 archive 选一个父节点 → 提出"interesting modification" → benchmark 实测 → 入库（不替换父节点，stepping stones 思想）
- **进化对象**：coding agent 自身代码（编辑工具、长上下文管理、peer-review 机制等）
- **冻结对象**：基础模型 + benchmark + 安全沙箱
- **headline**：SWE-bench 20.0% → 50.0%；Polyglot 14.2% → 30.7%
- **与 tree 的对应**：
  - DGM 的 archive ↔ tree 的 cell forest
  - DGM 的 "stepping stones" 思想 ↔ tree 的 op-log + cell ancestry
  - DGM 的 benchmark gate ↔ tree 的 漏斗裁决 + Ring 阈值
- **关键差异**：DGM 进化的是"agent 代码"，粒度=可执行程序；tree 进化的是"cell + 算符序列"，粒度=结构化知识单元。DGM 需要 LLM 写代码，tree 不需要——更便宜、更可审计
- **写论文时定位**：Related Work 第一段重点 baseline

### 4. JiuwenClaw
- **出处**：https://github.com/openJiuwen-ai/jiuwenclaw （openJiuwen-ai 团队，文档`docs/en/SkillSelfEvolution.md`）
- **核心组件**（综述里详细但 README 没明写，引自 datawhale 文章）：
  - **SignalDetector**：基于规则、不调用 LLM。监视工具错误信号、用户纠错措辞
  - **SkillEvolutionManager**：编排
  - **SkillOptimizer**：把信号写成条目落到 `evolutions.json`
  - **solidify**：合适时机把 `evolutions.json` 合并回 `SKILL.md`
- **failure → Troubleshooting 段；correction → Examples 段**
- **手动通道**：`/evolve` 命令
- **与 tree 的对应**：JiuwenClaw 的 `evolutions.json` ↔ tree 的 op-log（未固化的草稿）；solidify ↔ tree 的 crystallize；SkillOptimizer ↔ tree 的 after_step hook
- **关键差异**：JiuwenClaw 信号源是用户/工具显式信号（rule-based），tree 信号源是能量动力学（隐式连续）；JiuwenClaw 是单 skill 文件级别，tree 是 cell 网络级别（有 connect/quarantine/decay）
- **写论文时定位**：Related Work 第一段次重点；强调"信号源 explicit vs implicit"差异

### 5. EvoSkill
- **出处**：Alzubi/Provenzano/Bingham/Chen/Vu, arXiv 2603.02766（2026-03），https://github.com/sentient-agi/EvoSkill ，Sentient Labs
- **核心算法**：把 GEPA prompt-patching 思想从 prompt 扩到"整套 agent program"
  - 5 阶段循环：Base Agent run → **Proposer**（从失败提建议）→ **Generator**（写新 skill/prompt）→ **Evaluator**（held-out 数据打分）→ **Frontier**（Top-N 版本作 Git 分支 + `frontier/*` 标签）
- **模式**：`skill_only` / `prompt_only`；scorer 包括 rule match / LLM-as-judge / 执行脚本
- **核心声明**："Pareto frontier of agent programs governs selection, retaining only skills that improve held-out validation performance while the underlying model remains frozen"
- **headline**：
  - OfficeQA exact-match 60.6% → 67.9% (+7.3pt)
  - SealQA 26.6% → 38.7% (+12.1pt)
  - 零样本迁移：SealQA 上演化出的 skill 直接到 BrowseComp +5.3pt
- **与 tree 的对应**：
  - Proposer/Generator/Evaluator/Frontier 链路 ↔ tree 的 crystallize → connect → reference → quarantine
  - "model frozen, skill evolves" 与 tree 一致
  - Git 分支版本 ↔ tree 的 op-log replay
- **关键差异**：EvoSkill 离线 benchmark 驱动，tree 是 in-session 在线驱动；EvoSkill 粒度=整套 agent program，tree 粒度=cell 单元；EvoSkill 没有腐朽/能量动力学，靠 Pareto Frontier 显式剪枝
- **写论文时定位**：Related Work 第一段重点 baseline，引数字突出"skill-level evolution 可迁移"这一已被证实的论点，为我们的 tree-as-harness 做铺垫
- **跨域迁移那条结论尤其关键**：评 reviewer 时可以用来反驳"自进化只在训练集 overfit"

### 6. Ultron (ModelScope)
- **出处**：https://github.com/modelscope/ultron ，demo `writtingforfun-ultron.ms.show/dashboard` ，依赖 Twinkle
- **三 Hub 架构**：
  - **Memory Hub**：HOT/WARM/COLD 分层（按命中率百分位重平衡）+ L0/L1/Full 摘要 + RAG + Presidio 中英 PII 脱敏 + 时间衰减 `hotness = exp(-α·days)`
  - **Skill Hub**：从 hot memory 聚类晶化 → 多步工作流 skill；内置"provenance-grounded verification"和"structure-score upgrade gate"保证 evolved skill 不退化；与 ModelScope 80K+ 外部 skill 统一检索
  - **Harness Hub**：发布 persona + memory + skill 为可分享 blueprint（short-code 导入），双向 sync
- **特别项**：Trajectory Hub（2026-04-26）支持 SFT/self-training 闭环
- **与 tree 的对应**：
  - Ultron 的"三层蒸馏产物"（memory/skill/harness）↔ tree 的"cell / op-log / harness"
  - "provenance-grounded verification" ↔ tree 的能量来源约束
  - "structure-score upgrade gate" ↔ tree 的 Ring 阈值 + lignification 检查
  - 时间衰减 `exp(-α·days)` ↔ tree 的 decay 算符
- **关键差异**：Ultron 面向"群体共享"（跨 agent/用户），tree 面向"单 agent 内部"；Ultron 在 memory layer 演化，tree 在 harness layer 演化
- **写论文时定位**：Related Work 第二段借三层蒸馏叙事，但说清"tree is single-agent intrinsic, not collective infrastructure"

### 7. OpenSpace (HKUDS)
- **出处**：https://github.com/HKUDS/OpenSpace （港大数据科学组），无 arXiv 论文，按 README/Changelog
- **三类编辑操作**：
  - **AUTO-FIX**：skill 失败时即时修复（最小 diff）
  - **DERIVED**：从已有 skill 派生专门化子 skill（一个 `document-gen-fallback` 派生出 13 个变体）
  - **CAPTURED**：从成功执行轨迹捕获新 skill（44 个 file-format I/O skill 中 32 个 captured，29 个 execution-recovery skill 中 28 个 captured）
- **触发器**：Post-Execution Analysis（任务后） + Tool Degradation（工具成功率下降） + Metric Monitor（周期健康巡检）
- **谱系**：本地 Dashboard 提供"Version Lineage — Skill Evolution Graph"DAG 视图
- **外部服务**：MCP server（stdio / SSE / streamable HTTP 三种 transport）；云 registry 区分 public/team/private 可见性；CLI `openspace-download-skill` / `openspace-upload-skill`
- **质量保证**："Post-write verification"——按结构化 pattern 评每个新 skill 提交
- **与 tree 的对应**：
  - FIX/DERIVED/CAPTURED 三操作 ↔ tree 的 crystallize 三态（修补 / 派生 / 新生）
  - DAG lineage ↔ tree 的 op-log + cell ancestry（1:1 映射，可直接引）
  - 三类触发器 ↔ tree 的 after_step/after_episode 钩子
  - Post-write verification ↔ tree 的 needs_review + quarantine
- **关键差异**：OpenSpace 是 MCP 外挂服务，跨 host 共享；tree 是 OuterHarness 单 agent 内嵌；OpenSpace 没有能量动力学，靠成功率统计
- **写论文时定位**：Related Work 第二段重点对标版本化机制；DAG lineage 那条可作为"这类范式被社区认可"的引证

### 8. SkillClaw (AMAP-ML)
- **出处**：Ma/Yang/Ji/Wang/Wang/Hu/Huang/Chu, arXiv 2604.08377（2026-04），高德/AMAP ML 团队，https://github.com/AMAP-ML/SkillClaw
- **架构**：
  - **Client Proxy**：拦截 `/v1/chat/completions` 和 `/v1/messages`，记录 artifact，维护本地 skill 库
  - **Evolve Server**（可选）：从共享存储（本地/OSS/S3）读，生成或重写 skill；两种模式：
    - `workflow` 模式：Summarize → Aggregate → Execute 三段 pipeline
    - `agent` 模式：在 OpenClaw workspace 里编辑
  - 可选 PRM 跨 session 质量检查
- **目标受众**：跨 session/设备/用户/agent 共享技能
- **benchmark**：WildClawBench 上 Qwen3-Max 显著提升（abstract 没给具体数字）
- **与 tree 的差异**：和 SkillClaw 不在同一层——SkillClaw 关注"共享存储 + 代理拦截"的工程问题，tree 关注"单 agent 内部知识结构"的方法学问题
- **写论文时定位**：Related Work 第二段次要引用，不展开

### 9. OpenClaw-RL (Gen-Verse)
- **出处**：Wang/Chen/Jin/Wang/Yang, arXiv 2603.10165，https://github.com/Gen-Verse/OpenClaw-RL ，依赖 Tinker/Fireworks AI
- **架构**：server-client；RL server 在 inference API 后面 host policy，user terminal 通过 HTTP 流式回传交互数据；独立异步 server 抽信号，不阻塞 inference
- **两类信号统一**：
  - **Evaluative signals**（综述里叫 Binary RL / GRPO）：覆盖广、粒度粗
  - **Directive signals**（综述里叫 OPD = On-Policy Distillation）：token-level supervision、密集但稀疏
- **稳定性技巧**：overlap-guided hint selection（选 teacher 分布与 student top-k 重叠最大的 hint）+ log-prob-difference clip（限制 per-token advantage）
- **覆盖环境**：terminal / GUI / SWE / tool-call；personal agent + long-horizon setting
- **与 tree 的对应**：
  - 信号双流 ↔ tree 的能量动力学双轴
  - 异步训练 ↔ tree 的 OuterHarness 解耦（哲学一致，层次不同）
- **关键差异**：OpenClaw-RL 改的是 LoRA 权重（参数层），tree 改的是 cell/算符序列（结构层）。这两条是 orthogonal 而非互斥
- **写论文时定位**：Related Work 第三段 orthogonal axis；强调"我们这一层不依赖 reward model、不需要训练"

### 10. Agent Lightning (Microsoft)
- **出处**：Luo/Zhang/He/Wang/Zhao/Li/Qiu/Yang, arXiv 2508.03680 （2025-08，Microsoft Research），https://www.microsoft.com/en-us/research/project/agent-lightning/
- **核心机制**：把 agent 行为建模为 MDP；**LightningRL** 算法（含 credit assignment 模块）；**Training-Agent Disaggregation** 把训练 process 和 agent runtime 完全解耦，"almost ZERO code modifications"
- **集成**：LangChain / OpenAI Agents SDK / AutoGen / 裸 OpenAI client；多 agent 场景可只对选定 agent 做梯度
- **benchmark**：text-to-SQL / RAG / 数学工具调用，stable continuous improvement
- **与 tree 的对应**：
  - "执行和训练解耦"哲学 ↔ tree 的"OuterHarness 与 inner LLM 解耦"
  - 不同层次：Agent Lightning 解耦的是"agent runtime ↔ trainer"，tree 解耦的是"agent inner loop ↔ harness 算符层"
- **写论文时定位**：Related Work 第三段 orthogonal axis；引一句"decoupling philosophy is shared but our layer is data/operator, not gradient"

---

## 三、按 tree 四轴的对照矩阵

> 四轴：harness（外挂层是否存在）/ self-evolution（进化对象） / loop（反馈触发） / evaluation（量化指标）

| 方法 | harness | self-evolution | loop 触发 | evaluation |
|------|---------|---------------|-----------|------------|
| Hermes | 无外挂层（融在 agent 内） | memory + soul | reflection step | 隐式（用户满意度） |
| Agent Zero | 无（脚手架） | 开发者手改 | 无自动 | 无 |
| **DGM** | 有（archive） | agent 代码 | 每轮 benchmark | SWE-bench / Polyglot |
| JiuwenClaw | 无外挂层（融在 swarm 内） | SKILL.md | rule-based signal | 隐式 |
| **EvoSkill** | 有（Proposer/Generator/Evaluator/Frontier） | skill + prompt | 离线 benchmark | Pareto Frontier on held-out |
| **Ultron** | 有（三 Hub） | memory + skill + harness | 命中率衰减 + 轨迹回流 | provenance + structure-score gate |
| OpenSpace | 有（MCP 服务） | skill DAG | 三类触发器 | structural pattern score |
| SkillClaw | 有（Proxy + Evolve Server） | shared SKILL.md | API 拦截 + PRM | WildClawBench |
| OpenClaw-RL | 有（异步 server） | LoRA 权重 | 在线对话 PRM/Judge | 多环境 |
| Agent Lightning | 有（trainer disagg） | prompt / 权重 | spans 总线 | 多 benchmark |
| **tree-harness（ours）** | 有（OuterHarness 三 hook） | cell 网络 + 算符序列 | 能量动力学（隐式连续） | 漏斗裁决 + Ring 阈值 + SWE-bench Pro |

**tree 在矩阵里的独特位置**：harness 层 ✅ + 结构化进化 ✅ + 隐式连续触发（能量动力学）= 唯一一行；DGM 是结构化但显式 benchmark 触发；EvoSkill 是结构化但离线触发；Ultron 是结构化但群体目标；OpenSpace 是结构化但跨 host 服务。

---

## 四、Related Work 三段框架（按当前讨论结果落地）

**段一：Skill-level self-evolution within a frozen base model.**
对标 Darwin Gödel Machine、EvoSkill、JiuwenClaw。共同点：base model 冻结，演化对象是 agent 资产/skill。差异点：DGM/EvoSkill 离线 benchmark 驱动 + 粗粒度（整套 agent program），JiuwenClaw 在线 + rule-based signal + 单 skill 文件粒度，tree 在线 + 能量动力学 + cell 网络结构。

**段二：Versioned skill graphs and collective infrastructure.**
对标 OpenSpace、Ultron，SkillClaw 一句话带过。共同点：版本化 + lineage DAG + provenance。差异点：OpenSpace/Ultron 面向"跨 host/群体共享"，tree 面向"单 agent 内部结构"；OpenSpace 用成功率统计做 gate，tree 用能量动力学做 gate。

**段三：Parameter-level evolution as orthogonal axis.**
对标 OpenClaw-RL、Agent Lightning。强调：这两条作用在梯度层，tree 作用在结构层；二者可叠加而非互斥；tree 的优势是无需训练、无需 reward model、closed-form 可审计。

**Introduction 略过**：Hermes / Agent Zero 用一句"prior work treats this as memory or scaffolding; we reframe as harness with energy dynamics"带过即可。

---

## 五、TODO / 待补料

1. **EvoSkill 的 Pareto Frontier 实现细节**：abstract 没说清"Top-N"具体怎么选，需要读 PDF（arXiv 2603.02766 v1）确认是否多目标 NSGA-II 类
2. **DGM 的 archive growth 策略**：是否有 pruning？还是无限扩展？需要读论文 §3
3. **JiuwenClaw 的 SignalDetector 规则列表**：README 没给，需读 `docs/en/SkillSelfEvolution.md` 或 `jiuwenswarm/` 源码
4. **Ultron 的 "structure-score upgrade gate" 具体公式**：README 提了名字没给公式，需查 ms-agent 子库
5. **SkillClaw WildClawBench 的具体数字**：arXiv 2604.08377 PDF
6. **OpenClaw-RL 的 rollout filter / 健康带 [30%,60%]**：abstract 没提，需要读 PDF 看是否就是 MemAgent 那个健康带概念（mem 记忆里有这条）
7. **Agent Lightning 的 LightningStore / spans schema**：abstract 没展开，需要读 PDF §3
