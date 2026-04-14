import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("results/exp4_results.csv")
df = df[df["prediction"] != "unknown"]

print("=" * 55)
print("实验四结果：各类别检测难度对比")
print("=" * 55)

summary = []

for cat in ["face", "animal", "object", "nature"]:
    subset = df[df["category"] == cat]

    y_true = (subset["ground_truth"] == "fake").astype(int)
    y_pred = (subset["prediction"]   == "fake").astype(int)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # 分开看真图/假图各自的准确率
    real_sub  = subset[subset["ground_truth"] == "real"]
    fake_sub  = subset[subset["ground_truth"] == "fake"]
    real_acc  = (real_sub["prediction"] == "real").mean()
    fake_acc  = (fake_sub["prediction"] == "fake").mean()

    print(f"\n{cat.upper()}")
    print(f"  整体准确率：{acc:.1%}")
    print(f"  F1-Score：  {f1:.3f}")
    print(f"  真实图识别率：{real_acc:.1%}  "
          f"（{int(real_acc*len(real_sub))}/{len(real_sub)}）")
    print(f"  AI图识别率：  {fake_acc:.1%}  "
          f"（{int(fake_acc*len(fake_sub))}/{len(fake_sub)}）")

    summary.append({
        "Category":    cat,
        "Accuracy":    round(acc,  3),
        "F1-Score":    round(f1,   3),
        "Real Acc":    round(real_acc, 3),
        "Fake Acc":    round(fake_acc, 3),
        "Total":       len(subset)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/exp4_summary.csv", index=False)
print("\n\n汇总表：")
print(summary_df.to_string(index=False))