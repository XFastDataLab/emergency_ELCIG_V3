# 基于执行逻辑自洽的应急演练脚本迭代生成方法

本仓库提供 ELCIG（Execution-Logic Consistency-driven Iterative Generation）的实验代码、实验记录及参考资料。方法使用大语言模型生成脚本，通过时序、职责和资源规则进行验证，再根据反馈定向修正。

支持三个场景：`S1` 危险化学品泄漏、`S2` 台风灾害、`S3` 地质灾害。

## 1. 仓库结构

```text
.
├── README.md
├── 代码/emergency_ELCIG_V3/
│   ├── config.yaml          # 实验与服务配置
│   ├── .env.example         # 环境变量示例
│   ├── requirements.txt
│   ├── src/
│   │   ├── main.py          # 命令行入口
│   │   ├── experiments.py   # 实验组织、共享初稿、结果保存
│   │   ├── elcig.py         # 验证与修正循环
│   │   ├── evaluation.py    # 验证器测试样本及指标
│   │   ├── llm/            # 模型调用适配器
│   │   ├── rag/            # WeKnora 检索适配器
│   │   ├── rules/          # 本地约束与别名
│   │   └── validator/      # 时序、职责、资源验证器
│   ├── tests/              # 离线测试
│   └── output/             # 新运行产生的数据
├── 实验过程数据/
│   ├── 实验一：验证器有效性/
│   ├── 实验二：ELCIG曲线/
│   └── 实验三：消融实验/
├── 演练案例/
└── 预案/
```

历史记录位于 `实验过程数据/`；新命令的结果写入代码项目的 `output/`。现有资料文件不等同于可直接使用的 WeKnora 数据库，需要自行导入知识库并完成索引。

## 2. 安装环境

以下命令在 Windows PowerShell 中执行。先进入仓库根目录，再进入代码项目；后文命令均在代码项目目录运行。

```powershell
Set-Location './代码/emergency_ELCIG_V3'

# 创建自己的环境，不依赖仓库中可能保留的本机 .venv
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 避免中文控制台日志乱码
$env:PYTHONIOENCODING = 'utf-8'

# 查看支持的命令参数，不发起模型或检索请求
.\.venv\Scripts\python.exe -B -m src.main --help
```

上述示例使用 Python 3.13；其他版本需要自行验证依赖兼容性。Linux/macOS 可使用 `python3 -m venv .venv`，并将后文的 `.\.venv\Scripts\python.exe` 换为 `./.venv/bin/python`。依赖包括 `requests`、`pyyaml`、`python-dotenv`、`httpx`、`openai`；即使仅运行实验一，也需安装这些导入依赖。

## 3. 模型与知识库配置

实验一不调用模型或知识库；实验二和完整实验三需要可用的模型服务及 WeKnora 服务。单独运行 `LLM-only` 或 `Self-Refine` 只需要模型服务。

**当前适配器中仍有硬编码配置。仅填写 `.env` 或修改 `config.yaml` 的模型字段、知识库 ID，不会使所有设置自动生效。** 删除原凭据后，需要按下述位置接入自己的配置。下面的源码片段是需要自行应用的适配示例，不表示仓库已经完成这些修改。

将演练案例、应急预案、法律法规相关资料先自行上传到私有知识库

### 3.1 需要准备的信息

| 信息 | 用途 | 当前读取位置 |
| --- | --- | --- |
| 模型 API 地址 | 模型服务的接口根地址 | V4：`src/llm/dsutil.py` 的 `AIModelEnum.DeepSeek_V4` 的 `URL` |
| 模型 API Key | 模型调用鉴权 | V4：同一枚举的 `KEY` |
| 模型名称或部署 ID | 选择实际调用模型 | V4：同一枚举的 `MODEL` |
| WeKnora API 地址 | 例如 `http://localhost:8080/api/v1`，需包含 `/api/v1` | `config.yaml` 的 `rag.base_url` |
| WeKnora API Key | 检索接口鉴权 | `rag.api_key`，可由 `WEKNORA_API_KEY` 展开 |
| 知识库 ID | 指定已建好索引的知识库 | 当前写在 `src/rag/weknora.py` 的 `_hybrid_search()` 内，需按3.3节接入配置 |
| 模型请求参数 | 温度、超时、输出长度、思考模式等 | `src/llm/dsutil.py` 的实际请求函数 |

凭据只在本机配置，不提交 `.env`、真实密钥或包含凭据的源码。服务账号和知识库由使用者自行提供。

### 3.2 模型适配器

`src/llm/client.py` 根据 `llm.model` 是否包含 `v3` 选择调用分支：包含时调用 `call_deepseek_v3()`，其他名称调用 `call_deepseek_v4()`。它不会把配置中的完整模型名、温度和超时直接转发给底层请求。

对于 V4 分支，可将 `src/llm/dsutil.py` 中 `AIModelEnum` 内的 `DeepSeek_V4` 定义替换为以下形式（该文件已有 `import os`）：

```python
# 位于 class AIModelEnum(Enum): 内，保留相应缩进
DeepSeek_V4 = DEEPSEEK_V4(
    URL=os.getenv("LLM_BASE_URL", ""),
    KEY=os.getenv("DEEPSEEK_API_KEY", ""),
    MODEL=os.getenv("LLM_MODEL", ""),
    INPUT_MAXSIZE=100000,
)
```

在启动 Python **之前**，向当前 PowerShell 会话设置变量：

```powershell
# 替换为自己的服务信息；以下均为占位值
$env:LLM_BASE_URL = 'https://your-model-service.example/api/v3'
$env:LLM_MODEL = 'your-v4-model-or-deployment-id'
$env:DEEPSEEK_API_KEY = 'your-model-api-key'
```

这里要求先设置进程环境，是因为 `dsutil.py` 的枚举在模块导入时就会初始化，而 `main.py` 在后续执行 `main()` 时才加载 `.env`。仅把这些变量写入 `.env`，无法让上述导入时读取方式自动获得它们。

如使用 V3 分支，应配置同一文件中的 `VOLCES_BASE_URL`、`VOLCES_API_KEY`、`VOLCES_DPSK_V3`；也可将它们改为读取上述进程环境变量，并使 `config.yaml` 的 `llm.model` 包含 `v3`。只修改 `DeepSeek_V4_FLASH_OFFICAL` 不会改变当前同步 V4 调用路径。

当前 V4 请求函数 `call_deepseek_v4()` 设置 `max_tokens=16384`，通过 `extra_body` 禁用思考，未显式传入温度或超时。需要设置温度、超时或更换其他兼容服务时，应同步调整该函数和客户端构造；服务不支持的扩展字段需要按实际接口适配。配置中的 `temperature`、`timeout` 不能直接视为已生效的请求参数；`json_mode=True` 当前也只是本地调用标记，未转换为接口的 JSON 响应格式参数。

### 3.3 WeKnora 知识库

1. 准备可访问的 WeKnora 服务，创建知识库并导入实验所需资料。
2. 等待文档解析和检索索引完成，取得 API Key 和知识库 ID。
3. 在 `src/rag/weknora.py` 的 `WeKnoraRetriever.__init__()` 中启用知识库 ID 保存：

```python
self.knowledge_base_id = knowledge_base_id
```

4. 将 `_hybrid_search()` 中写死的 `kb_id = "..."` 替换为：

```python
kb_id = self.knowledge_base_id
```

当前实现没有启用多知识库参数传递，`knowledge_base_ids` 保持空数组；需要多库检索时，应另行实现并验证对应请求逻辑。

本适配器访问 `/knowledge-bases/{知识库ID}/hybrid-search`，先尝试 GET，失败再尝试 POST，请求体含 `query_text` 和 `match_count`。服务须与这一接口约定兼容。开启 RAG 后，检索失败或没有有效片段会终止该次实验。

### 3.4 环境变量与 YAML 示例

如果本机还没有 `.env`，复制模板后编辑；已有文件则直接编辑，避免覆盖已有配置。

```powershell
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
}
```

`.env` 中填写知识库配置：

```dotenv
WEKNORA_API_KEY=your-weknora-api-key
WEKNORA_BASE_URL=http://localhost:8080/api/v1
WEKNORA_KNOWLEDGE_BASE_ID=your-knowledge-base-id
```

在完成3.2、3.3节的适配后，将 `config.yaml` 中对应的 `rag` 和 `llm` 段修改为以下形式，保留其余实验配置：

```yaml
rag:
  use_rag: true
  provider: weknora
  base_url: "${WEKNORA_BASE_URL}"
  api_key: "${WEKNORA_API_KEY}"
  knowledge_base_id: "${WEKNORA_KNOWLEDGE_BASE_ID}"
  knowledge_base_ids: []
  top_k: 5
  max_context_chunks: 12
  resource_urls: handle

llm:
  base_url: "${LLM_BASE_URL}"
  api_key: "${DEEPSEEK_API_KEY}"
  model: "${LLM_MODEL}"
  temperature: 0.3   # 当前适配器未转发；需在实际请求函数中配置
  timeout: 90        # 当前适配器未转发；需在实际请求函数中配置
```

模型变量使用3.2节中提前设置的进程环境；知识库变量由 `main.py` 加载 `.env` 后展开。若当前进程已存在同名变量，默认的 `load_dotenv()` 不会覆盖它。

## 4. 主要实验参数

| 参数 | 当前含义 |
| --- | --- |
| `experiment.scenarios` | 批量实验场景列表，默认 `[S1, S2, S3]` |
| `experiment.max_iterations` | ELCIG 最大修正次数，默认5；0表示不修正 |
| `--repeats N` | 每个场景独立重复次数，优先于可选的 `experiment.repeats`，缺省1 |
| `rag.top_k` | 每条查询最多采用的检索片段数 |
| `rag.max_context_chunks` | 多路检索去重后的最大片段数 |
| `validator.responsibility_strict` | 当前配置为 `true`，未知职责组合按硬错误处理，仍保留代码中的词汇容错 |
| `validator.temporal_strict` | 当前配置为 `false`，缺失前置条件仅记为提示 |
| `action_grounding.use_llm` | 当前为 `false`，使用本地动作归一与匹配；启用后会增加模型调用 |
| `action_grounding.similarity_threshold` | 本地动作匹配阈值，默认0.34 |

`--method` 和 `--ablation` 会根据方法定义覆盖 `rag.use_rag`、`validator.use_validator`、`correction.mode`。Self-Refine 在当前实现中最多调用一次自反馈修正；`max_iterations=0` 时不调用。

`output.results_dir` 不控制当前实际输出位置；`experiment.save_intermediate` 不关闭当前历史保存逻辑。`validator_target_errors` 是旧配置字段，当前验证实验使用 `--samples`，旧 `--errors` 参数会报错。

## 5. 实验一：验证器有效性

该实验使用人工设计模板及确定性变换构造测试脚本和预期错误标注，不调用模型或知识库。数据属于受控构造测试，并非现场采集的独立专家标注样本。

```powershell
# 三个场景共36条模板样本，每个场景12条
.\.venv\Scripts\python.exe -B -m src.main --validator-test --samples 36

# 每个场景增加1条边界样本，共39条样本
.\.venv\Scripts\python.exe -B -m src.main --validator-test --samples 36 --include-boundary-case

# 只测试危化品场景：12条模板样本 + 1条边界样本
.\.venv\Scripts\python.exe -B -m src.main --validator-test --scenario S1 --samples 12 --include-boundary-case

# 额外打印完整预期错误标注
.\.venv\Scripts\python.exe -B -m src.main --validator-test --samples 36 --include-boundary-case --verbose
```

`--samples` 表示所有选定场景的模板样本总数，不包含追加的边界样本。模板样本数至少为“场景数×8”，加入边界样本后的总数不得超过100。检测率按“检出的预期错误实例数／全部预期错误实例数”计算，额外检测出的错误不增加该指标的分子。

输出包括每个场景的 `dataset.json`、`result.json`、`trace.jsonl`，以及启用边界测试时的 `boundary_sample.json`。汇总目录还包含 `comparison.csv`、`comparison.json`、`comparison.html` 和 `boundary_explanations.json`。

## 6. 实验二：ELCIG 迭代修正

先完成模型和知识库配置。以下示例对三个场景分别独立运行4次，最大修正次数由 `experiment.max_iterations` 控制。

```powershell
.\.venv\Scripts\python.exe -B -m src.main --method ELCIG --scenario S1 --repeats 4
.\.venv\Scripts\python.exe -B -m src.main --method ELCIG --scenario S2 --repeats 4
.\.venv\Scripts\python.exe -B -m src.main --method ELCIG --scenario S3 --repeats 4
```

先运行一个场景一次以检查接入：

```powershell
.\.venv\Scripts\python.exe -B -m src.main --method ELCIG --scenario S1 --repeats 1
```

每次重复重新生成初稿。`history.json` 中 `iteration=0` 表示初稿验证，后续整数表示已完成的修正次数。

绘制逻辑错误曲线时，读取各轮 `error_count`，或当前结果中的 `detected_error_curve`；读取各轮 `csr` 绘制约束满足率。当前 `error_curve` 采用 `pending_error_count`，可能包含旧的延后反馈，不等同于当前脚本的错误数量。命令负责生成数据，不自动生成论文图。

当前代码在首轮硬错误数超过50时随机选取部分反馈，其余与下一轮重检结果拼接，不去重；未提供命令行随机种子参数。历史数据、旧说明中的阈值和轮次可能不同，比较实验时需要保留对应代码及配置版本。

## 7. 实验三：消融对照

| 方法 | 初稿使用RAG | 修正方式 |
| --- | --- | --- |
| `LLM-only`（`LLM`） | 否 | 不修正 |
| `RAG-LLM`（`RAG`） | 是 | 不修正 |
| `Self-Refine`（`SelfRefine`） | 否 | 最多一次模型自反馈修正 |
| `ELCIG` | 是 | 规则反馈引导的迭代修正 |

所有方法的输出均由同一规则验证器收集评价指标；“不使用验证器修正”不表示不进行结果评价。

```powershell
# 四种方法 × 三个场景 × 1次独立重复，共12个方法结果
.\.venv\Scripts\python.exe -B -m src.main --ablation --repeats 1

# 四种方法 × 三个场景 × 5次独立重复，共60个方法结果
.\.venv\Scripts\python.exe -B -m src.main --ablation --repeats 5

# 仅场景S1，四种方法各重复2次
.\.venv\Scripts\python.exe -B -m src.main --ablation --scenario S1 --repeats 2
```

每个“场景×重复编号”生成两份初稿：LLM-only与Self-Refine共享无检索初稿，RAG-LLM与ELCIG共享有检索初稿。各分支独立复制脚本，修改不会影响另一分支。`--ablation` 不会附带运行实验一。

单独运行某一种方法的命令如下；这些独立命令不会相互共享初稿，成对消融比较应使用 `--ablation`。

```powershell
.\.venv\Scripts\python.exe -B -m src.main --method LLM-only --scenario S1 --repeats 1
.\.venv\Scripts\python.exe -B -m src.main --method RAG-LLM --scenario S1 --repeats 1
.\.venv\Scripts\python.exe -B -m src.main --method Self-Refine --scenario S1 --repeats 1
.\.venv\Scripts\python.exe -B -m src.main --method ELCIG --scenario S1 --repeats 1
```

当前RAG生成提示除检索内容外，还包含本地职责和时序规则；LLM-only组没有这两部分。因此该实现比较的是整套生成设置的差异，不能将差异全部解释为检索的独立作用。

## 8. 自定义配置与控制台日志

```powershell
# 复制配置后自行修改，例如最大修正次数、严格模式或检索参数
Copy-Item 'config.yaml' 'config.local.yaml'
.\.venv\Scripts\python.exe -B -m src.main --config config.local.yaml --method ELCIG --scenario S2 --repeats 1

# 同时显示控制台输出并保存到独立日志文件
$runLog = 'console-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log'
.\.venv\Scripts\python.exe -B -m src.main --ablation --repeats 1 2>&1 | Tee-Object -FilePath $runLog
```

自定义YAML应包含完整配置段；`--config` 不会自动将一个局部配置与默认文件合并。每次改变规则、模型或参数后，应保存该次实际配置及代码版本。

## 9. 输出文件与指标

新结果位于 `代码/emergency_ELCIG_V3/output/`，运行目录通常命名为 `场景-方法-MMdd-HHmm`，时间采用北京时间。同名目录存在时追加序号，不覆盖已有运行。

| 文件 | 内容 |
| --- | --- |
| `initial.json` / `final.json` | 结构化初稿和终稿 |
| `history.json` | 每轮脚本、检出错误、提示、受检约束数量及CSR |
| `result.json` | 方法、场景、重复编号、指标、曲线及调用耗时等 |
| `trace.jsonl` | 逐事件追加的生成、验证、修正和失败记录 |
| `shared_draft_reference.json` | 共享初稿标识、哈希及来源引用 |
| `shared_RAG_draft/` / `shared_noRAG_draft/` | 初稿副本、元数据、提示词与响应、检索记录 |
| `ALL-summary-*/results.json` | 本次命令的结果汇总 |

主要结果字段：

- `initial_errors`、`final_errors`：初稿、终稿的硬错误数量。
- `eredr`（兼容字段 `err`）：`(初始错误数−最终错误数)/初始错误数`。初始错误为0时字段保存0，`eredr_defined=false`，不应解释为有效的零改进测量。
- `csr`：当前实现为 `(受检约束数−硬错误数)/受检约束数`。提示不作为错误扣分，因此CSR为1不保证不存在提示或规则以外的问题。
- `iterations`：实际产生并记录的修正轮次数，不含初稿；调用没有改变脚本时，调用次数与该值可以不同。
- `draft_id`、`draft_hash`：用于核对消融实验的初稿复用关系。
- `generation_cost`、`refinement_cost`：调用次数、耗时、响应字符数；字符数不等同于服务端token用量。

LEDR用于有预期错误标注的验证器实验。对没有独立错误真值的生成脚本，不能仅凭验证器检出数量计算LEDR。

## 10. 离线测试与复现边界

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试使用预设响应或测试替身，不要求真实模型和知识库；部分测试会在 `output/ALL-tests-*` 下生成测试产物，这些文件应与正式实验数据区分。当前测试存在与实现阈值、轮次预期不一致的情况，应以实际测试输出为准，不能将历史说明中的通过数量视为当前保证。

实验命令展示的是当前代码的执行方式。历史归档可能来自不同代码或配置版本；重新请求外部模型和检索服务也不保证得到逐字相同的脚本。复现时应记录实际模型、接口请求参数、知识库版本、规则版本和全部独立重复结果。
