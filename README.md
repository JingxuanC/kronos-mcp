# Kronos MCP

清华开源 K 线时序基础模型 [Kronos](https://github.com/shiyu-coder/Kronos)
（论文 [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)，AAAI 2026）的独立
MCP（Model Context Protocol）HTTP 服务。零样本（zero-shot）输入任意 OHLCV
K 线序列，输出未来 open/high/low/close/volume 预测路径与交易视角信号，让任何
MCP 客户端（Claude Desktop、Kimi Code、Cursor、自研 Agent）直接调用金融市场
时序基础模型。

另接入亚马逊时序基础模型 **Chronos-2**（[amazon/chronos-2](https://huggingface.co/amazon/chronos-2)，
Apache-2.0，[`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting)
v2.x `Chronos2Pipeline`）作为可选后端：各预测工具传 `model="chronos2"` 即切换
（只预测收盘价 close 序列，volume 作为 past covariate；输出中位数路径 +
10%/90% 分位区间，不含 OHLC）。默认后端仍是 Kronos，行为完全不变。

模型代码与权重均为 **MIT** 协议（上游 `model/` 目录原样 vendor 进本仓，
许可证见 `LICENSE-Kronos`）；Chronos-2 为 **Apache-2.0**。本服务代码同样 MIT。

## 工具清单（5 个）

| 工具 | 说明 | 负载 |
|------|------|------|
| `forecast_kline` | 零样本 K 线预测：OHLCV 历史序列 → pred_len 根预测 K 线（OHLCV 均值路径）+ summary（方向/预期收益/预测波动率/耗时）。预测轴时间戳支持 `future_timestamps` 显式指定，否则按输入中位间隔自动顺延（日频/分钟频自适应），解析失败退回 index 序号并标注 | 中（同步） |
| `forecast_signal` | 交易视角信号：同一输入跑 N=min(sample_count,5) 次独立采样，统计终点收益方向一致率与离散度 → direction / expected_return_pct / confidence(0-1) / risk_note | 中（同步，比 forecast_kline 慢 N 倍） |
| `forecast_batch` | 批量预测：series_list 每项 {id, klines}，逐项容错（单项失败带 error 不拖垮整批）。入 JobQueue 异步执行，返回 job_id 轮询 `GET /jobs/<id>` | 重（异步） |
| `forecast_compare` | 双模型对比：同一输入依次跑 kronos + chronos2，返回两边预测路径、方向是否一致（`compare.directions_agree`）、各自预期收益。单边失败不拖垮另一边（该侧带 error） | 中偏重（同步，≈两模型耗时之和） |
| `model_info` | 当前已加载模型、参数量、device、max_context、内存/显存占用、HF 可用模型清单、chronos2 后端状态 | 轻（同步） |

约定：`lookback ≤ max_context(512)`，`pred_len ≥ 1`；`amount` 缺省时用
`volume*close` 近似。所有输出统一带 `method: "kronos-zero-shot"`
（forecast_compare 为 `"model-compare"`）。

**`model` 参数**（forecast_kline / forecast_signal / forecast_batch 通用，默认 `"kronos"`）：

- `"kronos"` 或缺省：Kronos 后端，预测完整 OHLCV；
- `"chronos2"`：Chronos-2 后端，只预测收盘价序列——预测条目中
  `close`=中位数路径、`open`=`close`、`high`/`low`=0.9/0.1 分位、`volume`=0，
  并加法式附加 `close_p10`/`close_p90` 字段；`summary.note` 有说明，
  `summary.pred_volatility` 为 10%-90% 区间宽度。chronos2 是确定性分位数
  预测（无采样随机性），`T`/`top_p`/`sample_count` 对其不生效；
  forecast_signal 下只跑 1 次，confidence 恒为 1、不代表不确定性；
- 向后兼容：传 Kronos 的 HF repo id（如 `NeoQuasar/Kronos-mini`）仍走
  kronos 并热切换模型。

## 快速开始

```bash
pip install -r requirements.txt
python3 server.py --port 50059
```

首次 forecast 调用时才从 HuggingFace 下载并加载模型（惰性加载，默认
`NeoQuasar/Kronos-small` 24.7M 参数）。国内网络可设镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

验证：

```bash
curl http://127.0.0.1:50059/health
curl http://127.0.0.1:50059/tools   # 应返回 5 个工具
```

接入 MCP 客户端（以 Claude Desktop / Kimi Code 为例）：

```yaml
# mcp 配置
kronos:
  url: http://127.0.0.1:50059/mcp
```

## 调用示例

### forecast_kline — K 线预测

```bash
curl -s http://127.0.0.1:50059/mcp -d '{
  "jsonrpc": "2.0", "id": 1, "method": "tools/call",
  "params": {"name": "forecast_kline", "arguments": {
    "klines": [{"timestamps": "2024-08-29 11:25:00", "open": 9.86, "high": 9.89,
                "low": 9.86, "close": 9.86, "volume": 625.0, "amount": 617074.0}, ...],
    "pred_len": 120, "lookback": 400, "sample_count": 5
  }}}'
```

返回（JSON 字符串）：

```json
{"method": "kronos-zero-shot",
 "predictions": [{"timestamps": "2024-08-29 11:30:00", "open": 9.87, "high": 9.90,
                  "low": 9.85, "close": 9.88, "volume": 512.0}, ...],
 "summary": {"last_close": 9.86, "pred_close_at_horizon": 9.92,
             "expected_return_pct": 0.61, "direction": "up",
             "pred_volatility": 0.35, "model": "NeoQuasar/Kronos-small",
             "device": "mps", "elapsed_ms": 18230},
 "timestamps_mode": "inferred"}
```

`timestamps_mode`：`provided`（用了 future_timestamps）/ `inferred`（按输入
中位间隔顺延）/ `index`（时间戳解析失败，退回序号，另带 `timestamps_note`）。

### forecast_signal — 交易信号

```bash
curl -s http://127.0.0.1:50059/mcp -d '{
  "jsonrpc": "2.0", "id": 2, "method": "tools/call",
  "params": {"name": "forecast_signal", "arguments": {
    "klines": [...], "pred_len": 10, "sample_count": 3
  }}}'
```

返回：

```json
{"method": "kronos-zero-shot", "model": "NeoQuasar/Kronos-small", "device": "mps",
 "direction": "up", "expected_return_pct": 0.42, "confidence": 0.75, "runs": 3,
 "sample_returns_pct": [0.45, 0.38, 0.43], "sign_consistency": 1.0,
 "return_std_pct": 0.036, "risk_note": "多次采样方向一致、离散度低，信号相对可靠（仍为统计预测，非投资建议）",
 "summary": {...}, "timestamps_mode": "inferred"}
```

confidence = 方向一致率 × 1/(1+收益std%)：采样方向越一致、离散越小越接近 1。

### forecast_batch — 批量（异步）

```bash
# 1) 提交 → 拿 job_id
curl -s http://127.0.0.1:50059/mcp -d '{
  "jsonrpc": "2.0", "id": 3, "method": "tools/call",
  "params": {"name": "forecast_batch", "arguments": {
    "series_list": [{"id": "sh600977", "klines": [...]},
                    {"id": "bad", "klines": [{"timestamps": "x"}]}],
    "pred_len": 20
  }}}'
# → {"job_id": "ab12cd34ef56", "status": "queued", "poll": "/jobs/ab12cd34ef56", ...}

# 2) 轮询取结果（单项失败不拖垮整批，带 error 字段）
curl -s http://127.0.0.1:50059/jobs/ab12cd34ef56
```

### forecast_compare — 双模型对比

```bash
curl -s http://127.0.0.1:50059/mcp -d '{
  "jsonrpc": "2.0", "id": 5, "method": "tools/call",
  "params": {"name": "forecast_compare", "arguments": {
    "klines": [...], "pred_len": 20
  }}}'
```

返回：

```json
{"method": "model-compare",
 "kronos":   {"predictions": [...], "summary": {...}, "timestamps_mode": "inferred"},
 "chronos2": {"predictions": [{"timestamps": "...", "open": 9.87, "high": 9.95,
                "low": 9.80, "close": 9.88, "volume": 0.0,
                "close_p10": 9.80, "close_p90": 9.95}, ...],
              "summary": {..., "note": "chronos2 只预测收盘价：..."}},
 "compare": {"directions_agree": true, "kronos_direction": "up",
             "chronos2_direction": "up", "kronos_expected_return_pct": 0.61,
             "chronos2_expected_return_pct": 0.44, "return_diff_pct": 0.17},
 "elapsed_ms": 41000}
```

某侧失败（如 chronos2 权重未放置）时该侧为 `{"error": "..."}`，`compare` 为 null。

### model_info

```bash
curl -s http://127.0.0.1:50059/mcp -d '{
  "jsonrpc": "2.0", "id": 4, "method": "tools/call",
  "params": {"name": "model_info", "arguments": {}}}'
# → {"loaded": true, "model": "NeoQuasar/Kronos-small", "param_count": 24691208,
#    "device": "mps", "max_context": 512, "available_models": [...], ...}
```

## 模型配置

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | 预测模型（另有 `Kronos-mini` 4.1M / `Kronos-base` 102.3M，均 MIT） |
| `KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` | K 线分词器 |
| `KRONOS_DEVICE` | `auto` | `auto` = cuda > mps > cpu；也可显式 `cpu`/`cuda`/`mps` |
| `MODEL_CACHE` | 空 | 模型快照本地目录（Docker 镜像内置 `/models` 预下载）；设置后优先读本地，配 `HF_HUB_OFFLINE=1` 可纯离线 |
| `HF_ENDPOINT` | 空 | 透传 huggingface_hub，国内设 `https://hf-mirror.com` |
| `CHRONOS_MODEL_PATH` | `/app/models-cache/chronos-2` | Chronos-2 权重**本地目录**（`model="chronos2"` 时懒加载，`device_map="cpu"`）。GFW 环境不访问 HuggingFace，需预先把 `amazon/chronos-2` 的 HF 快照放进该目录 |

也可在单次调用里传 `model` 参数热切换模型（与当前不一致时自动重载）。

## Docker 部署

镜像构建期**预下载模型**到 `/models`（torch 装 CPU 版控制体积），容器首次
forecast 离线命中本地缓存、秒级响应：

```bash
docker compose up -d        # 构建镜像 + 启动容器（首次构建约 5-10 分钟）
docker compose ps
```

国内构建加速（均为 build-arg，按需组合；compose 里也有注释样例）：

```bash
# Docker Hub 不可达 → 基础镜像走加速站；PyPI 走清华源；HF 走镜像站
docker compose build \
  --build-arg BASE_IMAGE=docker.1ms.run/library/python:3.11-slim \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg HF_ENDPOINT=https://hf-mirror.com
```

完全离线构建：往 `wheels/` 放一个 torch linux CPU wheel
（`pip download "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu
--only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.11 --no-deps`；
注意 PyPI 上 aarch64 的 torch≥2.14 会拖 GB 级 CUDA 依赖，别直接用 PyPI 版），
再往 `models-cache/` 放 HF 快照目录（`models--NeoQuasar--*`，从
`~/.cache/huggingface/hub/` 拷贝即可），构建全程零下载。
代理说明：`wheels/` 内置 PySocks/socksio 离线 wheel，Docker Desktop 注入的
socks5 代理（~/.docker/config.json proxies）在构建期可直接用。

**Chronos-2（可选后端）离线部署**：

1. 依赖：往 `wheels/` 放 `chronos_forecasting` 及其依赖闭包的 wheel
   （`transformers`、`tokenizers`、`safetensors`、`einops` 等），Dockerfile
   检测到 `chronos_forecasting-*.whl` 即离线安装，之后的
   `pip install -r requirements.txt` 因依赖已满足不再联网解析它：
   ```bash
   pip download "chronos-forecasting>=2" -d wheels/ \
     --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.11
   # 注意剔除拖入的 torch GPU wheel（torch 由 Dockerfile 单独装 CPU 版）
   ```
2. 权重：把 `amazon/chronos-2` 的 HF 快照放到宿主 `models-cache/chronos-2/`
   （bind-mount 到容器 `/app/models-cache/chronos-2`，即 `CHRONOS_MODEL_PATH`
   默认值；可用 `huggingface-cli download amazon/chronos-2 --local-dir` 下载后拷贝）。
   Chronos-2 懒加载，仅首次 `model="chronos2"` 请求时载入。
3. 资源：Chronos-2 约 120M 参数（fp32 ≈ 500MB 权重），CPU 推理时 RSS 峰值约
   1.5-2GB（与 Kronos 共存时）；3GB 内存机型可用但建议避免与 forecast_batch
   重任务并发，首次加载约需十几秒~1 分钟（CPU 反序列化），推理时
   `torch.set_num_threads(2)` 限线程。

换大模型：

```bash
docker compose build --build-arg KRONOS_MODEL=NeoQuasar/Kronos-base
# 或不重建镜像：docker compose run -e KRONOS_MODEL=NeoQuasar/Kronos-base ...
# （首次 forecast 时惰性下载；compose 里取消注释 kronos-models 卷可避免重下）
```

验证：

```bash
curl http://127.0.0.1:50059/health
curl http://127.0.0.1:50059/tools   # 应返回 5 个工具
```

license 鉴权（可选）：在 `docker-compose.yml` 中取消注释，把宿主机
`licenses.json` 挂进容器并设置 `MCP_LICENSE_FILE`。

## 与 Athena / 系列仓组合

```
astock-data-mcp / global-data-mcp   取 K 线（A股/全球行情）
        ↓ klines JSON
kronos-mcp (本仓 :50059)            Kronos 零样本预测 → 方向/预期收益/置信度
        ↓ 预测路径作为候选因子或信号
factor-miner-mcp (:50053)           因子回测/OOS 验证信号有效性
causal-mcp (:50057)                 事件研究/反事实验证信号因果性
```

示例：用 astock-data-mcp 拉 sh600977 的 5 分钟线 → 本仓 `forecast_signal`
得 direction/confidence → factor-miner-mcp `factor_backtest` 验证该信号在
历史上的 IC/收益表现。

## 端点一览

```
GET  /health        健康检查
GET  /tools         工具 JSON schema 列表
POST /mcp           MCP JSON-RPC（initialize / tools/list / tools/call）
GET  /jobs/<id>     异步任务状态/结果（forecast_batch）
GET  /quota         license 额度余量（鉴权模式）
GET  /queue-stats   队列概况
```

## 鉴权与额度（可选）

默认开放模式（本地/内网）。设置环境变量后强制 license key 鉴权：

```bash
export MCP_LICENSE_FILE=/path/to/licenses.json
python3 server.py --port 50059
# 客户端请求头：X-License-Key: <key>
```

license JSON 格式与额度语义见 `mcp_gateway.py` docstring（与
[factor-miner-mcp](https://github.com/JingxuanC/factor-miner-mcp) /
[causal-mcp](https://github.com/JingxuanC/causal-mcp) 相同）。
`GET /quota` 查余量，`GET /queue-stats` 看队列。重负载工具
（forecast_batch）计入 heavy_quota。

## Roadmap

- **Finetune 工具**：上游 [`finetune/`](https://github.com/shiyu-coder/Kronos/tree/master/finetune)
  目录支持基于 qlib 数据的微调（含 `finetune_csv` 自定义 CSV 管线），
  后续可封装为 `forecast_finetune` 异步工具（训练重负载，走 JobQueue）
- **Kronos-large**：上游预告 2026 Q1 发布更大模型，发布后 `KRONOS_MODEL`
  直接切换即可
- 更多市场适配：加密/期货高频线验证

## 致谢

- 模型与 `model/` 代码来自 [Kronos](https://github.com/shiyu-coder/Kronos)
  （MIT，vendor 自上游 commit `67b630e`，LICENSE 见 `LICENSE-Kronos`）
- 论文：Shi et al., "Kronos: A Foundation Model for the Language of
  Financial Markets", [arXiv:2508.02739](https://arxiv.org/abs/2508.02739),
  AAAI 2026；模型权重 [HuggingFace NeoQuasar](https://huggingface.co/NeoQuasar)（MIT）
- Chronos-2 后端：Ansari et al., "Chronos-2: From Univariate to Universal
  Forecasting", [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)；代码
  [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
  与权重 [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) 均为 Apache-2.0
- `mcp_gateway.py` 与 [factor-miner-mcp](https://github.com/JingxuanC/factor-miner-mcp) /
  [causal-mcp](https://github.com/JingxuanC/causal-mcp) 共用同一套鉴权/队列模块
- 测试数据 `examples/data/XSHG_5min_600977.csv` 来自上游 examples（600977
  上交所 5 分钟线，历史版本恢复，当前上游 master 已移除）

## License

MIT（本服务代码）；上游模型代码与权重同为 MIT（`LICENSE-Kronos`）。
