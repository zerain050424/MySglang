"""SessionRegistry 全局单例的 characterization tests。

钉住四件事：

1. `global_instance()` 两次返回同一对象（进程级共享，不是每实例一份）；
2. `reset_global()` 之后是新对象，旧对象内容仍在（不销毁调用方持有的引用）；
3. `clear()` 对共享实例生效 —— 一个持有者清空，其他持有者立刻看到空账本；
4. 缓存层的构造点走 `global_instance()`（源码断言，防止回归成 `SessionRegistry()`）。

    python test/manual/test_session_registry_global.py
"""

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))

from sglang.srt.mem_cache.uni_utils import SessionRegistry  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def test_singleton_identity():
    SessionRegistry.reset_global()
    a = SessionRegistry.global_instance()
    b = SessionRegistry.global_instance()
    check(a is b, "global_instance() must return the same object")


def test_reset_global_drops_instance():
    SessionRegistry.reset_global()
    a = SessionRegistry.global_instance()
    a.observe_request("s1", "r1", 100)
    check(len(a) == 1, "session must be registered")

    old = SessionRegistry.reset_global()
    check(old is a, "reset_global() must return the previous instance")

    b = SessionRegistry.global_instance()
    check(b is not a, "a new instance must be created after reset_global()")
    check(len(b) == 0, "the new instance must start empty")
    check(len(a) == 1, "the old instance must keep its own state")


def test_clear_is_shared():
    SessionRegistry.reset_global()
    a = SessionRegistry.global_instance()
    b = SessionRegistry.global_instance()
    a.observe_request("s2", "r2", 50)
    check(len(b) == 1, "second holder must see the same ledger")
    b.clear()
    check(len(a) == 0, "clear() must be visible to every holder")


def test_radix_cache_uses_global():
    src = HERE.parents[2] / "python/sglang/srt/mem_cache/radix_cache.py"
    text = src.read_text()
    check(
        "self.session_registry = SessionRegistry.global_instance()" in text,
        "radix_cache must use SessionRegistry.global_instance()",
    )
    check(
        "self.session_registry = SessionRegistry()" not in text,
        "radix_cache must not create a per-instance ledger",
    )


def main():
    tests = [
        test_singleton_identity,
        test_reset_global_drops_instance,
        test_clear_is_shared,
        test_radix_cache_uses_global,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILS.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    SessionRegistry.reset_global()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
