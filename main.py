import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "kumas_test_veri_toplama_sablonu.xlsx"
SHEET_NAME = "Veri Girişi"
OUTPUT_DIR = "graphs"

sns.set_theme(style="whitegrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FABRICS = ["İpek", "Pamuk", "Yün", "Kot"]
MODES = ["Mode 2", "Mode 4", "Mode 6", "Mode 8"]

def convert_to_binary(val):
    return 1 if str(val).strip() == "✓" else 0


def clean_label(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower().startswith("unnamed"):
        return ""
    return s


def get_predicted_fabric(raw_val, actual_fabric):
    if pd.isna(raw_val):
        return None

    text = str(raw_val).strip()

    if text == "✓":
        return actual_fabric

    for fabric in FABRICS:
        if text.lower() == fabric.lower():
            return fabric

    return None


df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=[1, 2])

df_raw = df_raw[df_raw.iloc[:, 0].notna()].copy()

user_series = df_raw.iloc[:, 0].copy()

data_raw = df_raw.iloc[:, 1:].copy()

cleaned_columns = []
for col in data_raw.columns:
    if isinstance(col, tuple) and len(col) >= 2:
        mode = clean_label(col[0])
        fabric = clean_label(col[1])
        cleaned_columns.append((mode, fabric))
    else:
        cleaned_columns.append((clean_label(col), ""))

data_raw.columns = pd.MultiIndex.from_tuples(cleaned_columns)


valid_cols = [
    c for c in data_raw.columns
    if isinstance(c, tuple) and c[0] in MODES and c[1] in FABRICS
]
data_raw = data_raw[valid_cols].copy()

data_bin = data_raw.copy()
for col in data_bin.columns:
    data_bin[col] = data_bin[col].apply(convert_to_binary)

print("Geçerli kolonlar:")
for c in data_raw.columns:
    print(c)

print(f"\nToplam kullanıcı sayısı: {len(user_series)}")


fabric_rows = []

for fabric in FABRICS:
    cols = [c for c in data_bin.columns if c[1] == fabric]
    if not cols:
        continue

    values = data_bin[cols].to_numpy().flatten()
    accuracy = float(values.mean() * 100)

    fabric_rows.append({
        "Fabric": fabric,
        "Accuracy": accuracy
    })

fabric_df = pd.DataFrame(fabric_rows)

plt.figure(figsize=(8, 5))
sns.barplot(data=fabric_df, x="Fabric", y="Accuracy")
plt.ylim(0, 100)
plt.title("Fabric Recognition Accuracy")
plt.xlabel("Fabric")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fabric_accuracy.png"), dpi=300)
plt.show()


mode_rows = []

for mode in MODES:
    cols = [c for c in data_bin.columns if c[0] == mode]
    if not cols:
        continue

    values = data_bin[cols].to_numpy().flatten()
    accuracy = float(values.mean() * 100)

    mode_rows.append({
        "Mode": mode,
        "Accuracy": accuracy
    })

mode_df = pd.DataFrame(mode_rows)

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=mode_df, x="Mode", y="Accuracy")

for i, mode in enumerate(mode_df["Mode"]):
    if mode == "Mode 6":
        ax.patches[i].set_facecolor("red")

plt.ylim(0, 100)
plt.title("Mode Accuracy Comparison")
plt.xlabel("Mode")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mode_accuracy.png"), dpi=300)
plt.show()

mode6_cols = [c for c in data_bin.columns if c[0] == "Mode 6"]

if mode6_cols:
    user_acc = data_bin[mode6_cols].mean(axis=1) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(user_acc) + 1), user_acc, marker="o")
    plt.ylim(0, 100)
    plt.title("Mode 6 Accuracy Across Users")
    plt.xlabel("User Index")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mode6_line_graph.png"), dpi=300)
    plt.show()
else:
    print("Mode 6 kolonları bulunamadı, line graph oluşturulamadı.")

actual_list = []
predicted_list = []

for _, row in data_raw.iterrows():
    for actual_fabric in FABRICS:
        col = ("Mode 6", actual_fabric)
        if col not in data_raw.columns:
            continue

        predicted = get_predicted_fabric(row[col], actual_fabric)
        if predicted is not None and predicted in FABRICS:
            actual_list.append(actual_fabric)
            predicted_list.append(predicted)

if actual_list and predicted_list:
    cm = pd.crosstab(
        pd.Series(actual_list, name="Actual"),
        pd.Series(predicted_list, name="Predicted"),
        dropna=False
    )
    cm = cm.reindex(index=FABRICS, columns=FABRICS, fill_value=0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Mode 6)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_mode6.png"), dpi=300)
    plt.show()
else:
    print("Confusion matrix için yeterli veri bulunamadı.")


group_rows = []

for mode in MODES:
    for fabric in FABRICS:
        col = (mode, fabric)
        if col not in data_bin.columns:
            continue

        accuracy = float(data_bin[col].mean() * 100)
        group_rows.append({
            "Mode": mode,
            "Fabric": fabric,
            "Accuracy": accuracy
        })

group_df = pd.DataFrame(group_rows)

plt.figure(figsize=(10, 6))
sns.barplot(data=group_df, x="Fabric", y="Accuracy", hue="Mode")
plt.ylim(0, 100)
plt.title("Accuracy by Mode and Fabric")
plt.xlabel("Fabric")
plt.ylabel("Accuracy (%)")
plt.legend(title="Mode")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grouped_modes_fabric.png"), dpi=300)
plt.show()


print("\nKaydedilen grafikler:")
for filename in [
    "fabric_accuracy.png",
    "mode_accuracy.png",
    "mode6_line_graph.png",
    "confusion_matrix_mode6.png",
    "grouped_modes_fabric.png",
]:
    path = os.path.join(OUTPUT_DIR, filename)
    print(f"- {path} -> {'OK' if os.path.exists(path) else 'YOK'}")