"""tools.py — Kronos K线时序预测 MCP 的工具注册表（4 个工具）。

封装清华开源 K 线基础模型 Kronos（https://github.com/shiyu-coder/Kronos，MIT，
论文 arXiv:2508.02739 / AAAI 2026）。上游 model/ 目录原样 vendor 进本仓。

- forecast_kline：零样本 K 线预测（同步，秒~分钟级，取决于 pred_len 与设备）
- forecast_signal：交易视角结论（方向 + expected_return + 多次采样置信度）
- forecast_batch：批量预测，重负载，入 JobQueue 异步执行（见 server.py ASYNC_TOOLS）
- model_info：当前加载模型 / 设备 / 参数量 / 可用模型清单

模型惰性加载：首次 forecast 才从 HuggingFace 下载 + 加载，线程锁防并发重复
加载。env：
    KRONOS_MODEL     预测模型 repo id（默认 NeoQuasar/Kronos-small）
    KRONOS_TOKENIZER 分词器 repo id（默认 NeoQuasar/Kronos-Tokenizer-base）
    KRONOS_DEVICE    cpu / cuda / mps / auto（默认 auto：cuda > mps > cpu）
    MODEL_CACHE      模型快照本地目录（Docker 镜像预下载到 /models）；
                     设置后优先走 snapshot_download 本地缓存，HF_HUB_OFFLINE=1
                     时纯离线命中；未设置则走默认 HF 缓存，挂卷换模型不受影响
    HF_ENDPOINT      透传给 huggingface_hub（如 https://hf-mirror.com）
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("kronos-mcp.tools")

METHOD = "kronos-zero-shot"
MAX_CONTEXT = 512  # 上游 max_context 上限（lookback 不得超过）
DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
# HF 上可用的官方模型清单（NeoQuasar 组织，全部 MIT）
AVAILABLE_MODELS = [
    {"id": "NeoQuasar/Kronos-mini", "params": "4.1M", "note": "最轻量，适合 CPU 快速试跑"},
    {"id": "NeoQuasar/Kronos-small", "params": "24.7M", "note": "默认，精度/速度平衡"},
    {"id": "NeoQuasar/Kronos-base", "params": "102.3M", "note": "最大开源版本，建议 GPU"},
]
# 方向判定阈值：|expected_return_pct| 小于该值视为 flat
FLAT_BAND_PCT = 0.1


# ── Toolkit interface（与 causal-mcp / factor-miner-mcp 相同）──
class ToolDef:
    def __init__(self, name: str, description: str, inputSchema: dict):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

    def to_dict(self):
        return {"name": self.name, "description": self.description, "inputSchema": self.inputSchema}


TOOLS: dict[str, ToolDef] = {}
HANDLERS: dict[str, callable] = {}


def tool(name: str, description: str, properties: dict, required: Optional[list] = None):
    """Decorator to register a tool."""
    def deco(fn):
        TOOLS[name] = ToolDef(name, description, {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        })
        HANDLERS[name] = fn
        return fn
    return deco


# ═══════════════════════════════════════════════════════════════
# 模型管理（惰性加载 + 线程锁）
# ═══════════════════════════════════════════════════════════════

class _ModelHolder:
    """Kronos predictor 单例。重依赖（torch/model）全部惰性导入。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        self._model = None
        self._model_name = None
        self._device = None

    @staticmethod
    def pick_device() -> str:
        want = os.environ.get("KRONOS_DEVICE", "auto").lower()
        if want != "auto":
            return want
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_repo_path(repo_id: str) -> str:
        """MODEL_CACHE 设置时走本地快照目录（构建期预下载），否则用 HF 默认缓存。"""
        cache = os.environ.get("MODEL_CACHE", "")
        if not cache:
            return repo_id
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, cache_dir=cache)
        logger.info("resolved %s from MODEL_CACHE: %s", repo_id, path)
        return path

    def get(self, model_name: Optional[str] = None):
        """返回 (predictor, model_name, device)。模型名与当前不一致时热切换。"""
        want = model_name or os.environ.get("KRONOS_MODEL", DEFAULT_MODEL)
        with self._lock:
            if self._predictor is not None and self._model_name == want:
                return self._predictor, self._model_name, self._device
            t0 = time.time()
            import torch  # noqa: F401 — 确保可用
            from model import Kronos, KronosTokenizer, KronosPredictor
            tok_id = os.environ.get("KRONOS_TOKENIZER", DEFAULT_TOKENIZER)
            logger.info("loading tokenizer %s + model %s ...", tok_id, want)
            tokenizer = KronosTokenizer.from_pretrained(self._resolve_repo_path(tok_id))
            model = Kronos.from_pretrained(self._resolve_repo_path(want))
            device = self.pick_device()
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
            self._predictor, self._model, self._model_name, self._device = predictor, model, want, device
            logger.info("model %s loaded on %s in %.1fs", want, device, time.time() - t0)
            return predictor, want, device


_HOLDER = _ModelHolder()


# ═══════════════════════════════════════════════════════════════
# 输入解析 / 校验
# ═══════════════════════════════════════════════════════════════

_TS_KEYS = ("timestamps", "timestamp", "date", "time", "datetime")


def _parse_klines(klines: list, lookback: Optional[int]):
    """klines JSON → (df, x_timestamp)。amount 缺省用 volume*close 近似。"""
    import numpy as np
    import pandas as pd
    if not isinstance(klines, list) or len(klines) < 2:
        raise ValueError("klines 至少需要 2 根K线")
    lb = lookback or len(klines)
    if lb < 2:
        raise ValueError("lookback 至少为 2")
    if lb > MAX_CONTEXT:
        raise ValueError(f"lookback 不能超过 max_context={MAX_CONTEXT}（当前 {lb}）")
    if len(klines) < lb:
        raise ValueError(f"klines 长度 {len(klines)} 小于 lookback {lb}")
    rows = klines[-lb:]
    ts_key = next((k for k in _TS_KEYS if k in rows[0]), None)
    recs = []
    for r in rows:
        try:
            o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"klines 元素缺少 open/high/low/close 或值非法: {e}")
        v = float(r.get("volume") or 0.0)
        a = r.get("amount")
        a = float(a) if a is not None else v * c  # amount 缺省 → volume*close 近似
        recs.append((r.get(ts_key) if ts_key else None, o, h, l, c, v, a))
    df = pd.DataFrame([r[1:] for r in recs],
                      columns=["open", "high", "low", "close", "volume", "amount"])
    ts_raw = [r[0] for r in recs]
    try:
        x_ts = pd.to_datetime(pd.Series(ts_raw))
        ts_ok = True
    except (ValueError, TypeError):
        x_ts = pd.Series(np.arange(len(recs)))
        ts_ok = False
    return df, x_ts, ts_ok


def _future_timestamps(x_ts, ts_ok: bool, pred_len: int, future_timestamps: Optional[list]):
    """预测轴时间戳：调用方给了就用；否则按输入中位间隔顺延（日/分钟自适应）；
    解析失败退回 index 序号并标注。"""
    import pandas as pd
    if future_timestamps:
        if len(future_timestamps) < pred_len:
            raise ValueError(f"future_timestamps 长度 {len(future_timestamps)} 小于 pred_len {pred_len}")
        return pd.to_datetime(pd.Series(future_timestamps[:pred_len])), "provided", None
    if not ts_ok:
        import numpy as np
        return pd.Series(np.arange(pred_len)), "index", "timestamps 解析失败，预测轴用 index 序号代替"
    diffs = x_ts.diff().dropna()
    if len(diffs) == 0:
        import numpy as np
        return pd.Series(np.arange(pred_len)), "index", "单点时间戳无法推断频率，用 index 序号代替"
    step = diffs.median()  # 中位间隔：5min 线的午休/跨日缺口不影响主频率
    y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])
    return y_ts, "inferred", None


def _fmt_ts(v) -> str:
    return v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, "strftime") else str(v)


# ═══════════════════════════════════════════════════════════════
# 核心预测
# ═══════════════════════════════════════════════════════════════

def _predict_core(klines, pred_len, lookback, future_timestamps,
                  T, top_p, sample_count, model_name):
    """跑一次预测，返回 (pred_df, y_ts, ts_mode, ts_note, model_name, device, elapsed_ms)。"""
    import pandas as pd  # noqa: F401
    if not isinstance(pred_len, int) or pred_len < 1:
        raise ValueError("pred_len 必须为 >= 1 的整数")
    df, x_ts, ts_ok = _parse_klines(klines, lookback)
    y_ts, ts_mode, ts_note = _future_timestamps(x_ts, ts_ok, pred_len, future_timestamps)
    predictor, mname, device = _HOLDER.get(model_name)
    t0 = time.time()
    with _HOLDER._lock:  # 推理串行化：CPU 场景下并发 predict 没有收益且容易打满
        pred_df = predictor.predict(df, x_ts, y_ts, pred_len=pred_len,
                                    T=T, top_p=top_p, sample_count=sample_count,
                                    verbose=False)
    elapsed_ms = int((time.time() - t0) * 1000)
    return pred_df, y_ts, ts_mode, ts_note, mname, device, elapsed_ms


def _summarize(pred_df, last_close: float, mname: str, device: str, elapsed_ms: int) -> dict:
    """预测结果 → 交易视角 summary。"""
    pred_close = float(pred_df["close"].iloc[-1])
    ret_pct = (pred_close / last_close - 1.0) * 100.0 if last_close else 0.0
    # 预测期波动：每根 (high-low)/close 的均值（%）
    rng = ((pred_df["high"] - pred_df["low"]) / pred_df["close"].clip(lower=1e-12)).mean() * 100.0
    direction = "up" if ret_pct > FLAT_BAND_PCT else ("down" if ret_pct < -FLAT_BAND_PCT else "flat")
    return {
        "last_close": round(last_close, 4),
        "pred_close_at_horizon": round(pred_close, 4),
        "expected_return_pct": round(ret_pct, 3),
        "direction": direction,
        "pred_volatility": round(float(rng), 3),
        "model": mname,
        "device": device,
        "elapsed_ms": elapsed_ms,
    }


def _paths_json(pred_df, y_ts) -> list:
    out = []
    for i in range(len(pred_df)):
        row = pred_df.iloc[i]
        out.append({
            "timestamps": _fmt_ts(y_ts.iloc[i]),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": round(float(row["volume"]), 2),
        })
    return out


_KLINES_PROP = {
    "type": "array",
    "description": "K线数组（时间升序）: [{timestamps, open, high, low, close, volume, amount?}, ...]。"
                   "timestamps 支持 ISO 字符串/日期；amount 缺省时用 volume*close 近似",
    "items": {"type": "object",
              "properties": {
                  "timestamps": {"type": "string"},
                  "open": {"type": "number"}, "high": {"type": "number"},
                  "low": {"type": "number"}, "close": {"type": "number"},
                  "volume": {"type": "number"}, "amount": {"type": "number"}},
              "required": ["timestamps", "open", "high", "low", "close"]},
}

_COMMON_PROPS = {
    "lookback": {"type": "integer", "description": f"回看长度（默认=klines 长度，上限 {MAX_CONTEXT}）"},
    "future_timestamps": {"type": "array", "items": {"type": "string"},
                          "description": "可选：预测轴时间戳（长度 >= pred_len）。不给则按输入中位间隔顺延推断"},
    "T": {"type": "number", "description": "采样温度（默认 1.0）", "default": 1.0},
    "top_p": {"type": "number", "description": "nucleus 采样阈值（默认 0.9）", "default": 0.9},
    "sample_count": {"type": "integer", "description": "并行采样条数（内部取均值路径，默认 5）", "default": 5},
    "model": {"type": "string",
              "description": "模型 repo id（默认 env KRONOS_MODEL 或 NeoQuasar/Kronos-small）"},
}


# ═══════════════════════════════════════════════════════════════
# 工具 1：forecast_kline（同步）
# ═══════════════════════════════════════════════════════════════

@tool("forecast_kline", "Kronos 零样本 K 线预测：输入 OHLCV 历史序列，输出 pred_len 根预测K线"
      "（open/high/low/close/volume）+ 交易视角 summary（方向/预期收益/预测波动率）。"
      "预测轴时间戳可用 future_timestamps 指定，否则按输入中位间隔自动顺延（日频/分钟频自适应）。"
      f"约束：lookback ≤ {MAX_CONTEXT}，pred_len ≥ 1。",
      {"klines": _KLINES_PROP,
       "pred_len": {"type": "integer", "description": "预测K线根数（>= 1）"},
       **_COMMON_PROPS},
      required=["klines", "pred_len"])
def forecast_kline(klines: list, pred_len: int, lookback: Optional[int] = None,
                   future_timestamps: Optional[list] = None, T: float = 1.0,
                   top_p: float = 0.9, sample_count: int = 5,
                   model: Optional[str] = None) -> str:
    pred_df, y_ts, ts_mode, ts_note, mname, device, ms = _predict_core(
        klines, pred_len, lookback, future_timestamps, T, top_p, sample_count, model)
    df_in, _, _ = _parse_klines(klines, lookback)
    last_close = float(df_in["close"].iloc[-1])
    result = {
        "method": METHOD,
        "predictions": _paths_json(pred_df, y_ts),
        "summary": _summarize(pred_df, last_close, mname, device, ms),
        "timestamps_mode": ts_mode,
    }
    if ts_note:
        result["timestamps_note"] = ts_note
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 工具 2：forecast_signal（同步，多次采样给置信度）
# ═══════════════════════════════════════════════════════════════

@tool("forecast_signal", "交易视角预测信号：对同一输入跑 N=min(sample_count,5) 次独立采样"
      "（每次 sample_count=1），统计终点收益的方向一致率与离散度，输出 direction / "
      "expected_return_pct / confidence(0-1) / risk_note。confidence = 方向一致率 × "
      "1/(1+收益std%)，多次采样方向越一致、离散越小越高。比 forecast_kline 慢 N 倍。",
      {"klines": _KLINES_PROP,
       "pred_len": {"type": "integer", "description": "预测K线根数（默认 10）", "default": 10},
       **_COMMON_PROPS},
      required=["klines"])
def forecast_signal(klines: list, pred_len: int = 10, lookback: Optional[int] = None,
                    future_timestamps: Optional[list] = None, T: float = 1.0,
                    top_p: float = 0.9, sample_count: int = 3,
                    model: Optional[str] = None) -> str:
    import numpy as np
    df_in, _, _ = _parse_klines(klines, lookback)
    last_close = float(df_in["close"].iloc[-1])
    runs = max(1, min(int(sample_count), 5))
    t0 = time.time()
    rets, preds = [], []
    mname = device = ""
    for _ in range(runs):
        pred_df, y_ts, ts_mode, ts_note, mname, device, _ = _predict_core(
            klines, pred_len, lookback, future_timestamps, T, top_p, 1, model)
        rets.append(float(pred_df["close"].iloc[-1]) / last_close - 1.0)
        preds.append(pred_df)
    rets_pct = np.array(rets) * 100.0
    mean_ret = float(rets_pct.mean())
    ret_std = float(rets_pct.std())
    n_up = int((rets_pct > FLAT_BAND_PCT).sum())
    n_down = int((rets_pct < -FLAT_BAND_PCT).sum())
    n_flat = runs - n_up - n_down
    direction, majority = ("up", n_up) if n_up >= n_down and n_up >= n_flat else \
        (("down", n_down) if n_down >= n_flat else ("flat", n_flat))
    consistency = majority / runs
    confidence = round(consistency / (1.0 + ret_std), 3)
    elapsed_ms = int((time.time() - t0) * 1000)
    if consistency >= 0.99 and confidence >= 0.6:
        risk_note = "多次采样方向一致、离散度低，信号相对可靠（仍为统计预测，非投资建议）"
    elif consistency < 0.6:
        risk_note = f"多次采样方向不一致（up={n_up}/down={n_down}/flat={n_flat}），信号可靠性低，谨慎使用"
    else:
        risk_note = f"方向基本一致但离散度偏高（std={ret_std:.2f}%），建议缩小仓位或等待确认"
    result = {
        "method": METHOD,
        "model": mname,
        "device": device,
        "direction": direction,
        "expected_return_pct": round(mean_ret, 3),
        "confidence": confidence,
        "runs": runs,
        "sample_returns_pct": [round(float(r), 3) for r in rets_pct],
        "sign_consistency": round(consistency, 3),
        "return_std_pct": round(ret_std, 3),
        "risk_note": risk_note,
        "summary": _summarize(preds[-1], last_close, mname, device, elapsed_ms),
        "timestamps_mode": ts_mode,
    }
    if ts_note:
        result["timestamps_note"] = ts_note
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 工具 3：forecast_batch（重负载 → JobQueue，server.py ASYNC_TOOLS）
# ═══════════════════════════════════════════════════════════════

@tool("forecast_batch", "批量 K 线预测（异步）：series_list 每项 {id, klines}，逐项容错"
      "（单项失败不拖垮整批，结果带 error 字段）。提交后返回 job_id，"
      "轮询 GET /jobs/<id> 取结果。每项输出同 forecast_kline 的 summary + predictions。",
      {"series_list": {"type": "array",
                       "description": "[{id: string, klines: [...]}, ...]，klines 格式同 forecast_kline",
                       "items": {"type": "object"}},
       "pred_len": {"type": "integer", "description": "预测K线根数（>= 1）"},
       **_COMMON_PROPS},
      required=["series_list", "pred_len"])
def forecast_batch(series_list: list, pred_len: int, lookback: Optional[int] = None,
                   future_timestamps: Optional[list] = None, T: float = 1.0,
                   top_p: float = 0.9, sample_count: int = 5,
                   model: Optional[str] = None) -> str:
    if not isinstance(series_list, list) or not series_list:
        raise ValueError("series_list 不能为空")
    t0 = time.time()
    results = []
    for item in series_list:
        sid = item.get("id", f"series-{len(results)}")
        try:
            pred_df, y_ts, ts_mode, ts_note, mname, device, ms = _predict_core(
                item["klines"], pred_len, lookback, future_timestamps,
                T, top_p, sample_count, model)
            df_in, _, _ = _parse_klines(item["klines"], lookback)
            results.append({
                "id": sid,
                "predictions": _paths_json(pred_df, y_ts),
                "summary": _summarize(pred_df, float(df_in["close"].iloc[-1]), mname, device, ms),
                "timestamps_mode": ts_mode,
                **({"timestamps_note": ts_note} if ts_note else {}),
            })
        except Exception as e:  # noqa: BLE001 — 单项容错
            logger.warning("forecast_batch item %s failed: %s", sid, e)
            results.append({"id": sid, "error": str(e)})
    return json.dumps({
        "method": METHOD,
        "total": len(series_list),
        "succeeded": sum(1 for r in results if "error" not in r),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "results": results,
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 工具 4：model_info
# ═══════════════════════════════════════════════════════════════

@tool("model_info", "当前已加载模型信息：模型名、参数量、device、max_context、内存/显存占用、"
      "可用模型清单。未加载（尚未 forecast）时返回 loaded=false 与默认配置。",
      {})
def model_info() -> str:
    import resource
    h = _HOLDER
    rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1) \
        if os.uname().sysname == "Darwin" else \
        round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    info = {
        "method": METHOD,
        "loaded": h._predictor is not None,
        "model": h._model_name or os.environ.get("KRONOS_MODEL", DEFAULT_MODEL),
        "tokenizer": os.environ.get("KRONOS_TOKENIZER", DEFAULT_TOKENIZER),
        "device": h._device,
        "max_context": MAX_CONTEXT,
        "memory_rss_mb": rss_mb,
        "available_models": AVAILABLE_MODELS,
        "model_cache": os.environ.get("MODEL_CACHE") or None,
    }
    if h._model is not None:
        info["param_count"] = sum(p.numel() for p in h._model.parameters())
        if h._device == "cuda":
            import torch
            info["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
            }
    return json.dumps(info, ensure_ascii=False)


EXTRA_SCHEMAS: dict = {}
