
## 任务目标：依据前驱知识点构建数据集的知识图谱

**实验方案简介**：利用学生的做题记录挖掘知识点之间的拓扑前驱关系，构建知识图谱；利用预测集做题记录验证图谱在下游任务（如认知诊断/知识追踪）中的预测效果。

---

## 核心算法流水线（Pipeline）

整个知识图谱拓扑边的构建遵循 **“基础数据聚合 $\rightarrow$ 相关性初筛 $\rightarrow$ 方向性深挖”** 的流水线。任何一个步骤未满足硬性条件，该知识点对 $(i, j)$ 将被立即丢弃，不再执行后续计算。

```
[步骤 1: 笛卡尔积统计] ──> 聚合出全局 a, b, c, d 频次
                                │
                                ▼
[步骤 2: 关联强度初筛] ──> 【硬性卡点】 φ ≥ 0.35 ？
                                │ Yes
                                ▼
[步骤 3: 方向判别定边] ──> 【硬性卡点】 样本频次卡点 (单点与联合成功数) ？
                                │ Yes
                                ▼
                           【硬性卡点】 正向、反向置信度与综合得分均 ≥ 0.6 ？
                                │ Yes
                                ▼
                        生成有效拓扑边 i -> j （i是j的前驱）

```

---

### 步骤 1：基础数据聚合（$a, b, c, d$ 统计量计算）

在真实数据集中，**知识点与题目通常是一对多**的关系。为了在知识点级别精准挖掘前驱关系，系统通过学生做题记录的**题目对（Question Pairs）笛卡尔积**来累加计算 $a, b, c, d$。

#### 1.1 核心统计逻辑

对于任意两个知识点 $i$ 和 $j$：

* **构建题目对**：对于同时做过这两个知识点相关题目的某位学生，将其做过的 $Q_i$ 题目子集与 $Q_j$ 题目子集进行两两组合（笛卡尔积），形成题目对集合：

$$Pair(i, j) = \{(q_m, q_n) \mid q_m \in Q_i \land q_n \in Q_j\}$$


* **状态映射**：遍历题目对，根据学生在这两道题上的实际得分（1为对，0为错），计入对应的计数器：
* $a = i\text{错} j\text{错}$
* $b = i\text{错} j\text{对}$
* $c = i\text{对} j\text{错}$
* $d = i\text{对} j\text{对}$


* **全局聚合**：遍历训练集中所有学生的做题记录，完成最终频次的全局叠加。

#### 1.2 边界条件与缺省规则

* **完全未接触则跳过**：若某学生**从未做过**知识点 $i$ 的任何题目，**或者**从未做过知识点 $j$ 的任何题目，该生贡献的题目对数量为 $0$，**完全不参与统计**。
* **部分做题动态收缩**：若学生仅做了部分题目，系统会动态收缩题目集，只对该学生实际产生了做题记录（0或1）的题目构建笛卡尔积。
* *示例*：$i$ 包含 3 题，$j$ 包含 3 题。某学生只做了 $i$ 的 1 道题和 $j$ 的 2 道题，则该生最终仅贡献 $1 \times 2 = 2$ 个题目对，而不是 $3 \times 3 = 9$ 个。



---

### 步骤 2：【第一道筛选】强相关性过滤（Phi 系数）

$\phi$ (Phi) 系数是专门用于 $2 \times 2$ 列联表的相关性指标，用来衡量两个二分类变量之间的绝对关联强度，用以**排除偶然巧合的伪关联边**。

#### 2.1 Phi 系数公式

$$\phi = \frac{ad - bc}{\sqrt{(a + b)(c + d)(a + c)(b + d)}}$$

* **物理含义**：$\phi$ 仅衡量“关联强度”，不区分谁是前驱、谁是后继（即无方向性，$\phi(i,j) = \phi(j,i)$）。取值范围 $[-1, 1]$，越接近 1 相关性越强。

#### 2.2 🚨 步骤 2 的硬性过滤条件

计算出 $\phi$ 后，必须立刻进行以下条件判定：

> * **IF $\phi \ge 0.35$**：判定为强相关，允许进入【步骤 3】进行方向深挖。
> * **ELSE ($\phi < 0.35$ 或为负数)**：判定为弱相关或负相关。**立即直接丢弃该知识点对**，阻断后续所有计算。
> 
> 

*注意：在实验调优阶段，可分别跑 `0.3`、`0.4`、`0.5` 三个阈值进行消融实验，对比下游模型效果选最优。*

---

### 步骤 3：【第二道筛选】方向判别与最终定边

对于通过步骤 2 筛选的强相关知识点对，引入具有**方向性**的正反向置信度公式与频次限制，确立前驱拓扑方向。

#### 3.1 🚨 步骤 3 的硬性前置条件（样本频次卡点）

在进行条件概率计算前，该知识点对必须**同时满足**以下两项稀疏性卡点，以防止极少样本引起的统计偏差：

> 1. **单个知识点成功样本足量**：`len(prereq_success) >= MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS`
> 2. **双知识点联合成功频次达标**：$d \ge MIN\_JOINT\_SUCCESSES\_FOR\_CORRECTNESS$
> 
> 

**以上任意一项不满足，该边直接过滤丢弃。**

#### 3.2 方向置信度计算与硬性截断

通过频次初筛后，计算并校验以下两个核心有向指标：

* **正向置信度**（做对后继 $j$ 的组合里，有多少也做对了前驱 $i$）：

$$P(I \mid J) = \frac{d}{b + d}$$



*原理：若 $i$ 是 $j$ 的前驱，掌握后继 $j$ 必然大概率能推出掌握前驱 $i$。*
> **硬性卡点**：必须满足 **$P(I \mid J) \ge 0.6$**，否则丢弃。


* **反向置信度**（做错前驱 $i$ 的组合里，有多少也做错了后继 $j$）：

$$P(\neg J \mid \neg I) = \frac{a}{a + b}$$



*原理：若前驱 $i$ 没掌握，后继 $j$ 理应大概率做错；若做错 $i$ 却能做对 $j$，说明 $i$ 不是 $j$ 的前驱。*
> **硬性卡点**：必须满足 **$P(\neg J \mid \neg I) \ge 0.6$**，否则丢弃。



#### 3.3 🚨 步骤 3 的终极硬性条件（综合得分判定）

融合正反向指标，通过**几何平均**计算最终的拓扑边权重：


$$\text{综合得分} = \sqrt[2]{P(I \mid J) \times P(\neg J \mid \neg I)}$$

最终的建边决策采用严格的**全过机制（AND 逻辑）**：

> **只有当满足以下终极硬性条件时：**
> * **正向置信度 $P(I \mid J) \ge 0.6$**
> * **且 反向置信度 $P(\neg J \mid \neg I) \ge 0.6$**
> * **且 综合得分 $\ge 0.6$**
> 
> 
> **决策**：判定该知识点对具备有效前驱关系，成功在知识图谱中构建一条有向边：**$i \longrightarrow j$**。综合得分将作为该边的权重，输入下游模型中。**反之，只要有一项低于 0.6，该边即被过滤丢弃。**



```

python generate_knowledge_graph/knowledge_graph_builder.py \
    --data-path datasets/moderate/MOOCRadar \
    --output-path datasets/knowledge_graph/MOOCRadar_knowledge_graph.json \
    --data-mode moderate \
    --train-split 1.0 \
    --exclude-first-n-students 50


python generate_knowledge_graph/knowledge_graph_builder.py \
    --data-path datasets/moderate/XES3G5M \
    --output-path datasets/knowledge_graph/XES3G5M_knowledge_graph.json \
    --data-mode moderate \
    --train-split 1.0 \
    --exclude-first-n-students 50
```
---

## 下游应用：基于知识图谱的 Few-Shot 样本选择策略（`KnowledgeGraphSelector`）

构建好的知识图谱（即 `knowledge_graph.json` 中 `is_prerequisite_for` 字段）被下游 `KnowledgeGraphSelector`（位于 `selection_strategies.py`）用于为测试题选取最相关的 few-shot 样本。该策略的核心思路是：**测试题可能包含多个知识点（Knowledge Point，以下简称 KP），对每个 KP 分别选样，最后按分层配额合并**。选样依据不仅看历史题是否包含测试知识点本身，也看是否包含其**前置知识点**（权重继承知识图谱中的 `correctness_score`）。

整个选择流水线遵循 **"构建前置索引 $\rightarrow$ 构建权重字典 $\rightarrow$ 逐知识点打分 $\rightarrow$ 分层抽样"** 的流程。任何一个阶段的数据缺陷都将触发对应的降级回退机制。

```
[测试题 skill_ids] ──> ┌──────────────────────────────────┐
                        │ 步骤 1：反转 is_prerequisite_for │
                        │ 得到 {概念 → {前驱: 权重}} 索引      │
                        └──────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │ 步骤 2：构建每 KP 权重字典         │
                        │ 自身 = 1.0 ，前驱 = correctness_score│
                        │ 每 KP 独立一份，互不干扰              │
                        └──────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │ 步骤 3：逐 KP 对所有历史题打分      │
                        │ 含本KP → 1.0                      │
                        │ 含前驱 → max(前驱权重)              │
                        │ 无关   → 0.0                      │
                        └──────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │ 步骤 4：分层抽样（三阶段）           │
                        │ ① 均分配额 + 贪心去重              │
                        │ ② 剩余候选跨组补位                 │
                        │ ③ 不足时随机填充                   │
                        └──────────────────────────────────┘
                                          │
                                          ▼
                                   按 index 排序输出
```

---

### 步骤 1：构建前置索引（Invert Prerequisite Relations）

知识图谱 JSON 中，每个概念的 `is_prerequisite_for` 字段表示 **"此概念是哪些概念的前驱"**（即此概念 → 后继）。例如：

```json
// 概念 "skill_1" 的 is_prerequisite_for：
// 含义：skill_1 是 skill_3、skill_5 的前驱（即 skill_1 → skill_3，skill_1 → skill_5）
"skill_1": {
  "is_prerequisite_for": {
    "skill_3": { "correctness_score": 0.82, ... },
    "skill_5": { "correctness_score": 0.75, ... }
  }
},
// 概念 "skill_2" 的 is_prerequisite_for：
// 含义：skill_2 也是 skill_3 的前驱（即 skill_2 → skill_3）
"skill_2": {
  "is_prerequisite_for": {
    "skill_3": { "correctness_score": 0.91, ... }
  }
}
```

为回答 **"给定概念 X，它的前驱是哪些？"** 这一问题（例如测试题考察了 skill_3，想找 skill_3 的前置知识），需要**反转**该映射关系。

反转逻辑如下：

```python
for prereq_id, node in concept_data.items():        # prereq_id = "skill_1"
    for dependent_id, data in node["is_prerequisite_for"].items():  # dependent_id = "skill_3"
        prereq_index[dependent_id][prereq_id] = score
```

得到如下索引结构 `{概念X → {前驱A: 权重, 前驱B: 权重}}`：

| 原始字段（图谱存储） | 含义 | 反转后的索引（前置索引） |
|---|---|---|
| `skill_1 → {"skill_3": 0.82, "skill_5": 0.75}` | skill_1 是 skill_3、skill_5 的前驱 | `skill_3 → {"skill_1": 0.82}`，`skill_5 → {"skill_1": 0.75}` |
| `skill_2 → {"skill_3": 0.91}` | skill_2 是 skill_3 的前驱 | `skill_3 → {"skill_2": 0.91}` |

最终生成的前置索引：

```
"skill_3" → {"skill_1": 0.82, "skill_2": 0.91}   （skill_3 的前驱是 skill_1(0.82) 和 skill_2(0.91)）
"skill_5" → {"skill_1": 0.75}                      （skill_5 的前驱是 skill_1(0.75)）
```

---

### 步骤 2：构建每个知识点的权重字典

对测试题中的**每个**知识点，独立构建一份权重字典：

| 来源 | 权重 |
|---|---|
| 该知识点自身 | **1.0**（固定） |
| 其直接前驱 | `correctness_score`（来自知识图谱），按分值降序取前 `max_related_concepts` 个 |

> 例：测试题含知识点 **i** 和 **j**，`max_related_concepts = 5`，前置索引如上：
> $$i \rightarrow \{i: 1.0,\ m: 0.9,\ n: 0.8\}$$
> $$j \rightarrow \{j: 1.0,\ p: 0.5,\ q: 0.6\}$$

> 🚨 **边界条件**：若某知识点在知识图谱中不存在或无其前驱，则权重字典退化为 `{自身: 1.0}`。

---

### 步骤 3：逐知识点对历史题打分

对测试题中每个知识点 KP$_k$，使用其权重字典 $W_k$ **独立**对该学生的所有历史题进行一次评分。对于一条历史题 $R$：

$$ \text{Score}_k(R) = \max_{\substack{c \in \text{skill\_ids}(R)}} \ W_k[c] $$

等价规则：

| 历史题包含的知识点 | 命中类型 | 分值 | 说明 |
|---|---|---|---|
| 包含该知识点自身 | 直接命中 | **1.0** | 该题直接考察了本知识点，满分 |
| 不含本知识点，但含其若干前驱 | 间接命中 | **max(前驱权重)** | 多个前驱取最高分，如 0.9 |
| 既不包含本知识点也不含其任何前驱 | 无关 | **0.0** | 不计分 |

> *例：测试知识点 **i** 的权重字典为 {i: 1.0, m: 0.9, n: 0.8}：*

| 历史题 skill_ids | 匹配过程 | 最终分数 |
|---|---|---|
| `["i"]` | max(1.0) = 1.0 | **1.0** |
| `["i", "x"]` | max(1.0, 0) = 1.0 | **1.0** |
| `["m"]` | max(0.9) = 0.9 | **0.9** |
| `["m", "n"]` | max(0.9, 0.8) = 0.9 | **0.9** |
| `["x", "y"]` | max(0, 0) = 0.0 | **0.0** |

> 🚨 **边界条件**：若历史题无 `skill_ids` 字段或为空数组，直接给 0 分。

一轮评分后，每组得到按分数降序排列的候选列表：

```
i 组: [(1.0, Q1), (1.0, Q2), (0.9, Q3), (0.9, Q4), (0.0, Q5), ...]
j 组: [(1.0, Q1), (0.6, Q4), (0.5, Q6), (0.5, Q7), (0.0, Q8), ...]
```

同分时按 `index` 升序排列。

---

### 步骤 4：分层抽样（Stratified Sampling with Dedup and Fill）

给定目标抽取数 `n_shots` 和测试知识点集合 $\{KP_1, KP_2, ..., KP_k\}$，分三阶段执行：

#### 4.1 🚨 第一阶段：均分配额 + 贪心去重

将 `n_shots` 均分到 k 个知识点，余数按顺序分配：

$$ \text{base} = \lfloor n_{\text{shots}} / k \rfloor,\quad \text{remainder} = n_{\text{shots}} \bmod k $$

$$ \text{quota}_i = \text{base} + \begin{cases} 1 & \text{if } i < \text{remainder} \\ 0 & \text{otherwise} \end{cases} $$

每个知识点从自己的候选列表中**按分数降序**依次选取，遇到已被其他知识点**选过的题则跳过**（贪心去重），直到填满配额或候选耗尽。

> **示例**（n_shots=8，KP={i, j}，配额均为 4）：
> ```
> i 组候选: [Q1(1.0), Q2(1.0), Q3(0.9), Q4(0.9), Q5(0.8), ...]
>           → 选中 Q1, Q2, Q3, Q4  ✓（配额满）
>
> j 组候选: [Q1(1.0), Q4(0.6), Q6(0.5), Q7(0.5), Q8(0.5), Q9(0.5), ...]
>           → Q1 已被选，跳过
>           → Q4 已被选，跳过
>           → 选中 Q6, Q7, Q8, Q9  ✓（配额满）
> ```
> 如果 j 组经过去重后只选到 3 条（配额余 1），进入第二阶段。

#### 4.2 🚨 第二阶段：跨组补位

第一阶段结束后，如果 `已选数量 < n_shots`，收集**所有组中未被选中**的候选记录，按分数降序排序，从最高分开始依次补选，直到补足差额：

```
第一阶段结果：已选 7 条，还差 1 条
所有组剩余候选: [(0.8, Q5), (0.5, Q10), (0.0, Q11), ...]
               → 选 Q5(0.8)  ✓
```

#### 4.3 🚨 第三阶段：随机填充

若所有组候选池均已耗尽（即没有任何历史题与任何测试知识点匹配），仍不足 `n_shots` 时，从**未被选中的全部历史记录**中随机抽取补齐：

```
仍需 2 条，无可用的高分候选
→ 从剩余历史记录中随机抽 2 条
```

#### 边界条件汇总

| 场景 | 行为 |
|---|---|
| `skill_ids` 为空 | 降级为 `RecentSelector`（取最近 n_shots 条） |
| records 为空 | 返回 `[]` |
| n_shots = 0 | 返回 `[]` |
| 某知识点无前驱 | 权重字典退化为 `{自身: 1.0}` |
| 历史题无 `skill_ids` | 该题在所有知识点组中均得 0 分 |
| 所有组候选合计 < n_shots | 第二阶段补位后，进入第三阶段随机填充 |

---

### 代码入口

```python
from selection_strategies import KnowledgeGraphSelector, create_selector

# 方式一：通过工厂函数
selector = create_selector(
    strategy="knowledge_graph",
    dataset_name="xes3g5m",
)

# 方式二：直接实例化
selector = KnowledgeGraphSelector(
    dataset_name="xes3g5m",            # 自动匹配知识图谱路径
    # 或指定路径:
    # graph_path="datasets/knowledge_graph/XES3G5M_knowledge_graph.json",
    max_related_concepts=5,            # 每个KP最多考虑的前驱数
)

selected = selector.select(
    records=historical_records,         # 学生历史做题列表
    n_shots=8,                          # 需要选取的样本数
    test_record=test_record,            # 当前测试题
)
# 返回按 index 升序排列的 selected records
```