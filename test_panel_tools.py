"""kronos 组形状容差验收：`_parse_klines` 是全部 5 个预测入口的唯一解析口，
所以在这里做形状容差即可覆盖 forecast_kline / forecast_signal /
forecast_compare / forecast_batch。

测试只碰解析层，不加载 566MB 模型 —— 快且无副作用。
"""

import numpy as np
import pytest

from tools import _parse_klines

N = 60
DATES = ["2026-06-%02d" % (1 + i % 28) for i in range(N)]
DATES = ["2026-%02d-%02d" % (6 + i // 28, 1 + i % 28) for i in range(N)]
rng = np.random.default_rng(7)
CLOSES = list(100.0 + np.cumsum(rng.normal(0, 1.0, N)))

BARS = [
    {"date": d, "open": c, "high": c * 1.01, "low": c * 0.99,
     "close": c, "volume": 1000 + i}
    for i, (d, c) in enumerate(zip(DATES, CLOSES))
]

SHAPES = {
    "flat_bars": BARS,
    "keyed_bars": {"sh600519": BARS},
    "columnar": {
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "rows": [[b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"]]
                 for b in BARS],
    },
    "wide_symbol_outer": {"sh600519": dict(zip(DATES, CLOSES))},
    "long_frame": [
        {"date": d, "symbol": "sh600519", "close": c} for d, c in zip(DATES, CLOSES)
    ],
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_parse_klines_shape_invariant(name):
    ref_df, ref_ts, ref_ok = _parse_klines(SHAPES["flat_bars"], None)
    df, ts, ok = _parse_klines(SHAPES[name], None)
    np.testing.assert_allclose(df["close"].to_numpy(), ref_df["close"].to_numpy(),
                               err_msg=name)
    assert list(df.columns) == list(ref_df.columns), name
    assert ok == ref_ok, name
    assert len(ts) == len(ref_ts), name


def test_lookback_still_works():
    df, _, _ = _parse_klines(SHAPES["keyed_bars"], 20)
    assert len(df) == 20
    np.testing.assert_allclose(df["close"].to_numpy(), CLOSES[-20:])


def test_bare_value_list_accepted():
    df, _, _ = _parse_klines(CLOSES, None)
    assert len(df) == N
    np.testing.assert_allclose(df["close"].to_numpy(), CLOSES)


def test_multi_symbol_rejected_with_actionable_message():
    """单序列工具收到多标的必须报错，而不是静默取第一个。"""
    multi = {"sh600519": BARS, "sz300750": BARS}
    with pytest.raises(ValueError) as e:
        _parse_klines(multi, None)
    assert "单序列" in str(e.value)


def test_too_short_rejected():
    with pytest.raises(ValueError):
        _parse_klines(BARS[:1], None)
    with pytest.raises(ValueError):
        _parse_klines({"nonsense": 1}, None)


def test_amount_defaults_to_volume_times_close():
    df, _, _ = _parse_klines([dict(b, amount=None) for b in BARS], None)
    np.testing.assert_allclose(
        df["amount"].to_numpy(), df["volume"].to_numpy() * df["close"].to_numpy()
    )
