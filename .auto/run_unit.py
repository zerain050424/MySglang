"""用 stub 顶替顶层包，跑纯 Python 的策略单测（本机无可用 torch/sglang）。"""
import importlib.util, sys, types

target = sys.argv[1]
for n in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
    m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = "python/sglang/srt/mem_cache/"
for mod in ("uni_utils", "victim_override", "evict_policy"):
    load(f"sglang.srt.mem_cache.{mod}", base + mod + ".py")

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("t", target)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
sys.exit(t.main())
