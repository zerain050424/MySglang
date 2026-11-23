"""仿真运行配置与淘汰策略参数装配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SimConfig:
    """单次仿真的可调配置项。"""

    capacity: int = 4096
    value_horizon_s: float = 120.0
    value_smoothing_k: float = 4.0
    value_floor_tokens: float = 0.0
    ageing_mode: str = "queue_ttl"
    session_ttl_k: float = 3.0
    session_ttl_min_s: float = 60.0
    session_ttl_max_s: float = 1800.0
    session_ttl_min_gaps: int = 2
    seed: int = 0
    page_size: int = 1
    cost_model: str = "uniform"
    verify: bool = True
    eviction_policy: str = "heter_cache"
    policy_config: Optional[dict] = None
    chain_policy: Optional[dict] = None
    regret_gate: Optional[dict] = None
    opt_use_map: Optional[str] = None
    opt_cost_map: str = ""
    opt_cost_mode: str = "protect_deep"
    idle_ttl: float = 3600.0
    oversize: str = "error"


def build_policy_config(cfg: SimConfig) -> Optional[Dict[str, Any]]:
    """根据策略族生成传给缓存工厂的参数字典。"""
    if cfg.eviction_policy == "heter_cache":
        config: Dict[str, Any] = {
            "share": {
                "ageing_mode": cfg.ageing_mode,
                "session_ttl_k": cfg.session_ttl_k,
                "session_ttl_min_s": cfg.session_ttl_min_s,
                "session_ttl_max_s": cfg.session_ttl_max_s,
                "session_ttl_min_gaps": cfg.session_ttl_min_gaps,
            },
            "value": {
                "horizon_s": cfg.value_horizon_s,
                "smoothing_k": cfg.value_smoothing_k,
                "floor_tokens": cfg.value_floor_tokens,
                "seed": cfg.seed,
            },
        }
        if cfg.chain_policy is not None:
            config["chain"] = cfg.chain_policy
        if cfg.regret_gate is not None:
            config["regret_gate"] = cfg.regret_gate
        return config

    if cfg.eviction_policy == "opt":
        if not cfg.opt_use_map:
            raise ValueError(
                "eviction_policy='opt' requires SimConfig.opt_use_map "
                "(see sim/opt_map.build_use_map)"
            )
        return {
            "use_map_path": cfg.opt_use_map,
            "block_size": cfg.page_size,
            "cost_map_path": cfg.opt_cost_map,
            "cost_mode": cfg.opt_cost_mode,
        }

    if cfg.policy_config is not None:
        return dict(cfg.policy_config)
    return None
