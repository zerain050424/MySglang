# HeterProfiler —— HeterCache 策略级观测账本 + profile 渲染器（方案 B 解耦）
#
# 定位：HeterCacheStrategy 的「记账 + 渲染」从策略类中剥离出的独立观测对象
#   1. 独占四本累计账：total_hit / total_evicted / total_evicted_one_shot /
#      victim_sources（键为裁决来源：one_shot 带 / weight:<queue> / weight:fallback）；
#   2. summary() 渲染 [HeterCache profile] 文本 —— 与旧 profile_summary
#      逐字节等价（验收基线：diff == 0）；
#   3. 输出口仅走 HeterLogger（heterlog 双写），本类不持文件句柄。
#
# 红线（本类绝不承载）：
#   - 树遍历（_per_queue_token_count 等仍在 RadixCache —— 它们的输出
#     是 profile 渲染的输入，属于策略数据面）；
#   - 背景采样线程（occ-sampler / node-label dump 留在 RadixCache）。
# 树状态由外部以参传入（queue_occ / session_summary），插件观测接口
# duck-typed（stats() / share_min），本类对策略零反向引用。
#
# 相对路径：本文件位于 Heter-Bench/profile/，向上一级到 Heter-Bench 再进 log。
# 级别门槛：渲染入口走 heterlog.info，级别语义由 HeterLogger 统一控制。
import os
import sys

_PROFILE_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log"))
if _PROFILE_PARENT not in sys.path:
    sys.path.insert(0, _PROFILE_PARENT)

from heter_logger import heterlog  # noqa: E402


class HeterProfiler:
    """HeterCache 策略观测账本（每实例一份；reset 与缓存 flush 同源）。

    与 strategy 的契约：
      record_hit / record_evict / record_victim_source —— 三个记账收口由
      策略调用，本类只做累加，不做任何判定；
      summary(queue_occ, session_summary, slru_policy, weights) —— 纯渲染，
      参数为只读引用。
    """

    def __init__(self):
        # profile 用的累计量（原 strategy 字段搬移，键=队列名）
        self._total_hit_tokens = {}
        self._total_evicted_tokens = {}
        self._total_evicted_one_shot = 0.0
        # 淘汰来源记账：one_shot 带 / 配额点名队列 / 随机兜底，
        # 供 profile 打印本次淘汰由哪个队列的策略决定。
        self._victim_sources = {}
        # profile 渲染走 heterlog 的 info 通道（级别门槛在 heterlog）
        self._heterlog = heterlog

    # ------------------------------------------------------------------
    # 记账收口（与 evict_policy 原实现逐行等价，仅状态换了持有人）
    # ------------------------------------------------------------------
    def record_hit(self, queue, hit_tokens):
        if self._heterlog.enabled:
            self._heterlog.debug("ProfileBook", f"hit {queue} +{hit_tokens}")
        self._total_hit_tokens[queue] = (
            self._total_hit_tokens.get(queue, 0.0) + hit_tokens
        )

    def record_evict(self, queue, tokens, one_shot):
        """one-shot 淘汰量单记标量；random 整行 one-shot。"""
        self._total_evicted_tokens[queue] = (
            self._total_evicted_tokens.get(queue, 0.0) + tokens
        )
        if one_shot:
            self._total_evicted_one_shot += tokens

    def record_victim_source(self, source, tokens):
        """记录本次淘汰由谁决定（one_shot 硬优先带 / 配额点名队列 /
        插件的 queue 名）。只做观测，不参与仲裁。"""
        self._victim_sources[source] = (
            self._victim_sources.get(source, 0.0) + tokens
        )

    def reset(self):
        self._total_hit_tokens.clear()
        self._total_evicted_tokens.clear()
        self._total_evicted_one_shot = 0.0
        self._victim_sources.clear()

    # ------------------------------------------------------------------
    # 渲染（summary —— 原 profile_summary 逐字节搬移）
    # ------------------------------------------------------------------
    def summary(self,
                queue_occ,
                session_summary=None,
                slru_policy=None,
                multi_agentic_policy=None,
                sched_occ=None,
                weights=None) -> str:
        """返回各队列的 profile 摘要。

        字段含义：
          hit / evicted : 累计命中 / 淘汰 token，按队列（与 occ 同口径，可比较）
          occ           : 该队列的全部驻留 token（含 response 段）；
                          `occ(-N)` 后缀 = 其中 one-shot 部分（random 整行、
                          struct 的 response 段）
          sched / w     : sched 是账本口径的可调度占用（排除 one-shot，含
                          locked）；w 是该队列的跨队列权重，淘汰顺序按
                          ``sched / w`` 降序。

        Σocc == schedulable + Σone_shot。
        多轮负载的 one_shot 恒为 0，不打印后缀；random 整行 one-shot。
        """
        from sglang.srt.mem_cache.heter_utils import QUEUE_NAMES
        weighted = sched_occ is not None
        header = "  queue              hit     evicted             occ"
        if weighted:
            header += "        sched       w"
        lines = [header]
        pool = 0
        schedulable = 0
        for q in QUEUE_NAMES:
            c = queue_occ.get(q)
            if not c:
                continue
            pool += c["total"]
            schedulable += c["total"] - c["one_shot"]
            hit = self._total_hit_tokens.get(q, 0.0)
            evicted = self._total_evicted_tokens.get(q, 0.0)
            occ_s = f"{c['total']:>10.0f}"
            if c["one_shot"]:
                occ_s += f"(-{c['one_shot']:.0f})"
            row = f"  {q:8s} {hit:>12.0f} {evicted:>12.0f} {occ_s:>16s}"
            if weighted:
                if weights is not None and q in weights:
                    row += (
                        f" {(sched_occ or {}).get(q, 0.0):>11.0f}"
                        f" {weights[q]:>7.2f}"
                    )
                else:
                    # random 不参与权重（整行 one-shot）
                    row += f" {'-':>11s} {'-':>7s}"
            lines.append(row)
        lines.append(
            f"  pool={pool:.0f} schedulable={schedulable:.0f} "
            f"one_shot_evicted={self._total_evicted_one_shot:.0f}"
        )
        if slru_policy is not None:
            # 共享会话数账本（SLRU 不参与选点，没有 picks 可报）。
            s = slru_policy.stats()
            lines.append(
                f"  slru: min_share={slru_policy.share_min} "
                f"share_counted={s['share_counted']:.0f} "
                f"share_dedup={s['share_dedup']:.0f} "
                f"aged_evicted={s['aged_evicted']:.0f}"
            )
        if self._victim_sources:
            src = " ".join(
                f"{k}={v:.0f}" for k, v in self._victim_sources.items()
            )
            lines.append(f"  victim_src: {src}")
        if multi_agentic_policy is not None:
            # agentic 带内复用分改写器：picks=0 说明窗口内没有 QAGENT victim。
            s = multi_agentic_policy.stats()
            lines.append(
                f"  multi_agentic: picks={s['ma_picks']:.0f} "
                f"fallback={s['ma_fallbacks']:.0f}"
            )
        if session_summary:
            # 会话级缺口：d_sum 与淘汰总量同量纲，可互相校验
            lines.append(
                f"  sessions={int(session_summary['sessions'])} "
                f"d_sum={session_summary['d_sum']:.0f} "
                f"d_p50={session_summary['d_p50']:.0f} "
                f"d_max={session_summary['d_max']:.0f} "
                f"d_nonzero={int(session_summary['d_nonzero'])}"
            )
        return "\n".join(lines)
