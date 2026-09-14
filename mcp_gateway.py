#!/usr/bin/env python3
"""域 MCP 服务的鉴权 + 额度 + 异步任务队列。

设计目标（内测 50 人规模，刻意保持零外部依赖）：

- LicenseStore：license key 鉴权 + 每日额度。key 清单是挂载进来的 JSON 文件
  （账号/License 服务签发后写入即可，域服务定时 reload 或重启生效）。
  未配置文件 → 开放模式（本地开发/内网部署行为不变）。
- JobQueue：重负载工具（回测/因子执行）异步化。提交即返回 job_id，
  客户端轮询 GET /jobs/<id>。线程池 + 队列上限 + 结果 TTL，
  防止多用户并发把 CPU 打爆。

都不引入 redis/DB：usage 落 JSON 文件（原子写），队列是纯内存
（进程重启丢队列是可接受的——任务幂等，客户端重新提交即可）。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("mcp-gateway")


def _int_env(name: str, default: int) -> int:
    """读整型环境变量，非法值回退默认（不因配置手误起不来）。"""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r 不是整数，回退默认 %d", name, raw, default)
        return default


# ═══════════════ License 鉴权 + 额度 ═══════════════

class QuotaExceeded(Exception):
    """当日额度用尽。message 面向用户，含剩余信息。"""


class LicenseStore:
    """license key 清单 + 每日用量计数。

    key 文件格式（MCP_LICENSE_FILE 指向）::

        {"keys": {"ak_xxx": {"name": "张三", "daily_quota": 200, "heavy_quota": 3}}}

    daily_quota：当日全部 tools/call 次数上限；heavy_quota：当日重负载
    任务（回测类）上限。0/缺省 = 不限。usage 落盘到同目录 .usage-<domain>.json。
    """

    def __init__(self, license_file: str = "", domain: str = "factor",
                 usage_dir: str = ""):
        self._file = Path(license_file).expanduser() if license_file else None
        # usage 落盘目录：MCP_USAGE_DIR > 参数 > license 文件同目录
        # （license 常以 ro 挂载进容器，不能往同目录写）
        base = (Path(usage_dir) if usage_dir else
                Path(os.environ.get("MCP_USAGE_DIR", "")) if os.environ.get("MCP_USAGE_DIR")
                else self._file.parent if self._file else None)
        self._usage_file = base / f".usage-{domain}.json" if base else None
        self._mtime = 0.0
        self._keys: dict[str, dict] = {}
        self._mu = threading.Lock()
        self._usage: dict[str, dict[str, dict[str, int]]] = {}  # date → key → counters
        if self._file:
            self.reload(force=True)
            self._load_usage()

    @property
    def enabled(self) -> bool:
        """配置了 key 文件且至少有一个 key → 强制鉴权；否则开放模式。"""
        return self._file is not None and len(self._keys) > 0

    def reload(self, force: bool = False):
        """license 文件 mtime 变了才重读（账号服务签发新 key 后免重启生效）。"""
        if not self._file or not self._file.exists():
            return
        mtime = self._file.stat().st_mtime
        if not force and mtime == self._mtime:
            return
        try:
            data = json.loads(self._file.read_text())
            keys = data.get("keys", {})
            with self._mu:
                self._keys = keys
                self._mtime = mtime
            logger.info("license keys reloaded: %d keys", len(keys))
        except Exception as e:  # noqa: BLE001
            logger.error("license file reload failed: %s", e)

    def check(self, key: str) -> tuple[bool, str]:
        """鉴权。开放模式一律放行；否则 key 必须在清单里。"""
        if not self.enabled:
            return True, ""
        self.reload()
        with self._mu:
            info = self._keys.get(key)
        if info is None:
            return False, "invalid or missing license key（请求头 X-License-Key）"
        return True, info.get("name", "")

    def consume(self, key: str, heavy: bool) -> dict:
        """扣一次额度，返回用量快照。超额抛 QuotaExceeded。"""
        if not self.enabled:
            return {"mode": "open"}
        self.reload()
        today = time.strftime("%Y-%m-%d")
        with self._mu:
            info = self._keys.get(key, {})
            bucket = self._usage.setdefault(today, {}).setdefault(
                key, {"calls": 0, "heavy": 0}
            )
            calls_left = self._left(info.get("daily_quota", 0), bucket["calls"])
            heavy_left = self._left(info.get("heavy_quota", 0), bucket["heavy"])
            if calls_left is not None and calls_left <= 0:
                raise QuotaExceeded(
                    f"当日调用额度已用完（{info.get('daily_quota')} 次/日），明天重置或联系管理员"
                )
            if heavy and heavy_left is not None and heavy_left <= 0:
                raise QuotaExceeded(
                    f"当日重负载任务（回测/挖掘）额度已用完（{info.get('heavy_quota')} 次/日）"
                )
            bucket["calls"] += 1
            if heavy:
                bucket["heavy"] += 1
            snapshot = {
                "calls_today": bucket["calls"],
                "heavy_today": bucket["heavy"],
                "calls_left": self._left(info.get("daily_quota", 0), bucket["calls"]),
                "heavy_left": self._left(info.get("heavy_quota", 0), bucket["heavy"]),
            }
        self._save_usage()
        return snapshot

    @staticmethod
    def _left(quota: int, used: int):
        return None if quota <= 0 else quota - used

    def quota_of(self, key: str) -> dict:
        """查询剩余额度（不扣减）。"""
        if not self.enabled:
            return {"mode": "open"}
        self.reload()
        today = time.strftime("%Y-%m-%d")
        with self._mu:
            info = self._keys.get(key, {})
            bucket = self._usage.get(today, {}).get(key, {"calls": 0, "heavy": 0})
            return {
                "name": info.get("name", ""),
                "calls_today": bucket["calls"],
                "heavy_today": bucket["heavy"],
                "calls_left": self._left(info.get("daily_quota", 0), bucket["calls"]),
                "heavy_left": self._left(info.get("heavy_quota", 0), bucket["heavy"]),
            }

    def _load_usage(self):
        if not self._usage_file or not self._usage_file.exists():
            return
        try:
            with self._mu:
                self._usage = json.loads(self._usage_file.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warn("usage file load failed, reset: %s", e)

    def _save_usage(self):
        if not self._usage_file:
            return
        try:
            # 只保留近 3 天，防文件膨胀
            with self._mu:
                days = sorted(self._usage)[-3:]
                slim = {d: self._usage[d] for d in days}
                self._usage = slim
            tmp = self._usage_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(slim, ensure_ascii=False))
            os.replace(tmp, self._usage_file)
        except Exception as e:  # noqa: BLE001
            logger.warn("usage save failed: %s", e)


# ═══════════════ Prometheus 指标 ═══════════════

class Metrics:
    """进程内 Prometheus 指标（纯标准库，线程安全）。

    延迟只记 sum/count 两个 counter（够算平均值），不引入 histogram 桶；
    /metrics 端点直接调 render() 输出 text exposition 格式。
    """

    def __init__(self):
        self._start = time.time()
        self._mu = threading.Lock()
        self._calls: dict[tuple[str, str], int] = {}   # (tool, status) → count
        self._lat_sum: dict[str, float] = {}           # tool → 累计秒数
        self._lat_count: dict[str, int] = {}           # tool → 计时次数
        self._license_checks: dict[str, int] = {}      # result → count

    def inc_call(self, tool: str, status: str):
        """status ∈ ok/error/rejected_license/rejected_quota/queued"""
        with self._mu:
            k = (tool, status)
            self._calls[k] = self._calls.get(k, 0) + 1

    def observe_latency(self, tool: str, seconds: float):
        with self._mu:
            self._lat_sum[tool] = self._lat_sum.get(tool, 0.0) + seconds
            self._lat_count[tool] = self._lat_count.get(tool, 0) + 1

    def inc_license_check(self, result: str):
        """result ∈ ok/invalid"""
        with self._mu:
            self._license_checks[result] = self._license_checks.get(result, 0) + 1

    def render(self, queue: "JobQueue | None" = None) -> str:
        """Prometheus text exposition 格式；传入 JobQueue 时追加队列指标。"""
        with self._mu:
            calls = sorted(self._calls.items())
            lat_sum = dict(self._lat_sum)
            lat_count = dict(self._lat_count)
            lic = sorted(self._license_checks.items())
        lines = [
            "# HELP mcp_tool_calls_total Total MCP tool calls by tool and status.",
            "# TYPE mcp_tool_calls_total counter",
        ]
        for (tool, status), n in calls:
            lines.append(f'mcp_tool_calls_total{{tool="{tool}",status="{status}"}} {n}')
        lines += [
            "# HELP mcp_tool_latency_seconds_sum Total seconds spent executing tools.",
            "# TYPE mcp_tool_latency_seconds_sum counter",
        ]
        for tool in sorted(lat_sum):
            lines.append(f'mcp_tool_latency_seconds_sum{{tool="{tool}"}} {lat_sum[tool]:.6f}')
        lines += [
            "# HELP mcp_tool_latency_seconds_count Number of timed tool executions.",
            "# TYPE mcp_tool_latency_seconds_count counter",
        ]
        for tool in sorted(lat_count):
            lines.append(f'mcp_tool_latency_seconds_count{{tool="{tool}"}} {lat_count[tool]}')
        lines += [
            "# HELP mcp_license_check_total License key checks by result.",
            "# TYPE mcp_license_check_total counter",
        ]
        for result, n in lic:
            lines.append(f'mcp_license_check_total{{result="{result}"}} {n}')
        lines += [
            "# HELP mcp_uptime_seconds Process uptime in seconds.",
            "# TYPE mcp_uptime_seconds gauge",
            f"mcp_uptime_seconds {time.time() - self._start:.1f}",
        ]
        if queue is not None:
            st = queue.stats()
            lines += [
                "# HELP mcp_queue_depth Current number of jobs waiting in queue.",
                "# TYPE mcp_queue_depth gauge",
                f"mcp_queue_depth {st['queue_size']}",
                "# HELP mcp_queue_jobs_total Async jobs finished by status.",
                "# TYPE mcp_queue_jobs_total counter",
            ]
            for status in ("done", "error", "expired"):
                n = st.get("jobs_total", {}).get(status, 0)
                lines.append(f'mcp_queue_jobs_total{{status="{status}"}} {n}')
        return "\n".join(lines) + "\n"


METRICS = Metrics()


# ═══════════════ 异步任务队列 ═══════════════

class QueueFull(Exception):
    """队列已满，客户端应稍后重试。"""


class JobTimeout(Exception):
    """单任务执行超过 job_timeout_sec（worker 侧判定，置 status=error / error=timeout）。"""


class JobQueue:
    """有界队列 + 固定 worker 线程池执行重负载工具。

    - 队列满 → 提交即拒（QueueFull），不让请求堆积
    - **单任务超时** job_timeout_sec（env MCP_JOB_TIMEOUT，默认 1800s，<=0 关闭）：
      handler 在一次性守护线程里执行，超时即置 status=error / error="timeout" 并
      放行 worker —— 挂死任务不再永久占位。注意 Python 无法强制中断线程，底层
      计算线程可能仍在后台跑到自然结束，其返回值被丢弃。
    - 结果保留 ttl_sec 供轮询；终态超 TTL 后转 "expired" 墓碑（job_id 语义可查），
      墓碑再保留 2×ttl 后清理
    - **reaper 回收 running 超时**：running 超过 running_timeout（默认 2×job_timeout，
      即 real worker 正常路径不会触发的上界）仍未完 → 判定 worker 失联，
      置 status=error 并回收
    - 每 key 同时只允许 1 个重任务在跑/排队（防单用户霸占总线）
    """

    def __init__(self, handlers: dict, workers: int = 2, maxsize: int = 50,
                 ttl_sec: int = 3600, job_timeout_sec: int | None = None,
                 reap_interval_sec: int = 300):
        self._handlers = handlers
        self._q: queue.Queue[str] = queue.Queue(maxsize=maxsize)
        self._jobs: dict[str, dict] = {}
        self._expired: dict[str, dict] = {}  # job_id → 过期墓碑（status="expired"）
        self._jobs_total = {"done": 0, "error": 0, "expired": 0}  # 累计数（给 /metrics）
        self._ttl = ttl_sec
        # 单任务超时：CLI/构造参数优先，否则 env MCP_JOB_TIMEOUT，默认 1800s
        self._job_timeout = (_int_env("MCP_JOB_TIMEOUT", 1800)
                             if job_timeout_sec is None else int(job_timeout_sec))
        self._running_timeout = self._job_timeout * 2 if self._job_timeout > 0 else 0
        self._reap_interval = max(1, int(reap_interval_sec))
        self._mu = threading.Lock()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"jobworker-{i}")
            for i in range(max(1, workers))
        ]
        for t in self._threads:
            t.start()
        threading.Thread(target=self._reaper, daemon=True, name="jobreaper").start()

    @property
    def job_timeout_sec(self) -> int:
        """单任务超时（秒）；<=0 表示不限制。"""
        return self._job_timeout

    def submit(self, tool: str, args: dict, key: str = "") -> str:
        # 单 key 并发闸：已有 queued/running 的重任务 → 拒绝
        with self._mu:
            for j in self._jobs.values():
                if j["key"] == key and key and j["status"] in ("queued", "running"):
                    raise QueueFull("你已有一个重任务在执行/排队中，等它跑完再提交")
        job_id = uuid.uuid4().hex[:12]
        with self._mu:
            self._jobs[job_id] = {
                "id": job_id, "tool": tool, "key": key, "status": "queued",
                "created_at": time.time(), "started_at": None,
                "finished_at": None, "result": None, "error": None,
            }
        try:
            self._q.put_nowait((job_id, tool, args))
        except queue.Full:
            with self._mu:
                del self._jobs[job_id]
            raise QueueFull("任务队列已满，请稍后重试")
        return job_id

    def get(self, job_id: str, key: str = "") -> dict | None:
        with self._mu:
            job = self._jobs.get(job_id)
            if job is None:
                # 墓碑：终态超 TTL 被回收，仍能告诉客户端"结果已过期"而非 not_found
                tomb = self._expired.get(job_id)
                if tomb is None:
                    return None
                if key and tomb["key"] and tomb["key"] != key:
                    return None
                return dict(tomb)
            if key and job["key"] and job["key"] != key:
                return None  # 看不到别人的任务
            out = dict(job)
        if out["status"] in ("done", "error") and out["finished_at"]:
            out["elapsed_sec"] = round(out["finished_at"] - out["created_at"], 1)
        return out

    def stats(self) -> dict:
        with self._mu:
            by_status = {}
            for j in self._jobs.values():
                by_status[j["status"]] = by_status.get(j["status"], 0) + 1
            expired = len(self._expired)
        return {"workers": len(self._threads), "queue_size": self._q.qsize(),
                "jobs": by_status, "expired_tombstones": expired,
                "job_timeout_sec": self._job_timeout,
                "jobs_total": dict(self._jobs_total)}

    def _run_guarded(self, tool: str, args: dict):
        """在一次性守护线程里跑 handler 并施加单任务超时。

        返回 (result, exc, timed_out)。超时后底层线程无法被强制中断，其返回值
        被丢弃（worker 已放行，任务已置 error）。
        """
        box: dict = {}

        def _run():
            try:
                box["result"] = self._handlers[tool](**args)
            except BaseException as e:  # noqa: BLE001 — 跨线程带回原始异常
                box["exc"] = e

        th = threading.Thread(target=_run, daemon=True, name=f"jobrun-{tool}")
        th.start()
        th.join(self._job_timeout if self._job_timeout > 0 else None)
        if th.is_alive():
            return None, None, True
        return box.get("result"), box.get("exc"), False

    def _worker(self):
        while not self._stop.is_set():
            try:
                job_id, tool, args = self._q.get(timeout=1)
            except queue.Empty:
                continue
            with self._mu:
                if job_id not in self._jobs:
                    self._q.task_done()
                    continue
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = time.time()
                created_at = self._jobs[job_id]["created_at"]
            try:
                result, exc, timed_out = self._run_guarded(tool, args)
                if timed_out:
                    raise JobTimeout(
                        f"timeout: 任务执行超过 {self._job_timeout}s 未完成，已置 error 并"
                        "放行 worker（底层计算线程无法被强制中断，可能仍在后台收尾）")
                if exc is not None:
                    raise exc
                with self._mu:
                    j = self._jobs.get(job_id)
                    # reaper 可能已把失联任务判 error（status != running）→ 不覆盖
                    if j is not None and j["status"] == "running":
                        j.update(status="done", result=str(result), finished_at=time.time())
                        self._jobs_total["done"] += 1
                METRICS.inc_call(tool, "ok")
            except BaseException as e:  # noqa: BLE001
                logger.error("job %s (%s) failed: %s", job_id, tool, e)
                with self._mu:
                    j = self._jobs.get(job_id)
                    if j is not None and j["status"] == "running":
                        j.update(status="error", error=str(e), finished_at=time.time())
                        self._jobs_total["error"] += 1
                METRICS.inc_call(tool, "error")
            finally:
                METRICS.observe_latency(tool, time.time() - created_at)
                self._q.task_done()

    def _reaper(self):
        while not self._stop.is_set():
            if self._stop.wait(self._reap_interval):
                break
            now = time.time()
            cutoff = now - self._ttl
            running_cutoff = now - self._running_timeout if self._running_timeout > 0 else None
            with self._mu:
                # 1) running 超时（worker 线程失联/异常退出遗留）→ 置 error 回收
                stuck = [(jid, j["tool"]) for jid, j in self._jobs.items()
                         if j["status"] == "running" and running_cutoff is not None
                         and (j.get("started_at") or j["created_at"]) < running_cutoff]
                for jid, _tool in stuck:
                    j = self._jobs[jid]
                    j.update(status="error",
                             error=(f"timeout: running 超过 {self._running_timeout}s 未完成，"
                                    "reaper 判定 worker 失联并回收"),
                             finished_at=now)
                    self._jobs_total["error"] += 1
                # 2) 终态超 TTL → 转 expired 墓碑（job_status 返回 expired 而非 not_found）
                dead = [jid for jid, j in self._jobs.items()
                        if j["status"] in ("done", "error")
                        and (j["finished_at"] or 0) < cutoff]
                for jid in dead:
                    j = self._jobs.pop(jid)
                    self._expired[jid] = {
                        "id": jid, "tool": j["tool"], "key": j["key"],
                        "status": "expired", "created_at": j["created_at"],
                        "finished_at": j["finished_at"],
                        "error": j.get("error"), "result": None,
                        "note": (f"结果已过期（默认保留 {self._ttl}s），如需结果请重新提交任务"),
                    }
                    self._jobs_total["expired"] += 1
                # 3) 墓碑再留 2×TTL 防内存膨胀
                tomb_cutoff = now - max(self._ttl * 2, 60)
                gone = [jid for jid, t in self._expired.items()
                        if (t.get("finished_at") or 0) < tomb_cutoff]
                for jid in gone:
                    del self._expired[jid]
            for _jid, tool in stuck:
                METRICS.inc_call(tool, "error")
            if stuck or dead:
                logger.info("reaper: %d running→error (超时回收), %d finished→expired",
                            len(stuck), len(dead))

    def shutdown(self):
        self._stop.set()
