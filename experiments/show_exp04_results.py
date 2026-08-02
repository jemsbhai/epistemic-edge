import json

path = "experiments/results/batadal_llm/04_batadal_llm_bonsai-1.7B_20260416_150858.json"  # run from repo root
with open(path, encoding="utf-8") as f:
    d = json.load(f)

print(f"Total trials: {d['total_trials']}")
print(f"Windows: {d['n_attack_windows']} attack, {d['n_normal_windows']} normal")
print()
print(f"{'Cond':<5} {'n':<5} {'JSON':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}  F1 95% CI")
print("-" * 58)
for c in ["A", "B", "C", "D", "E1", "E2", "F", "G"]:
    m = d["condition_summaries"].get(c)
    if m:
        ci = f"[{m['f1_ci_lower']:.3f}, {m['f1_ci_upper']:.3f}]"
        print(f"{c:<5} {m['n']:<5} {m['json_compliance']:>5.2f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  {ci}")
