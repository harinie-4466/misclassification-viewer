import pandas as pd

df = pd.read_csv("point_predictions.csv")

# Classification categories
CATEGORIES = ['CG', 'CN', 'NG', 'GN', 'NC', 'GC', 'CC', 'GG', 'NN']

# Label logic
def classify(expected, predicted):
    if expected.startswith("Coconut"):
        if predicted.startswith("green"): return "CG"
        elif predicted.startswith("non-green"): return "CN"
        elif predicted.startswith("Coconut"): return "CC"
    elif expected.startswith("green"):
        if predicted.startswith("non-green"): return "GN"
        elif predicted.startswith("Coconut"): return "GC"
        elif predicted.startswith("green"): return "GG"
    elif expected.startswith("non-green"):
        if predicted.startswith("green"): return "NG"
        elif predicted.startswith("Coconut"): return "NC"
        elif predicted.startswith("non-green"): return "NN"
    return "Unknown"

df['Classifications'] = df.apply(lambda row: classify(row['expected'], row['predicted']), axis=1)

# Count
grouped = df.groupby(['filename', 'Classifications']).size().reset_index(name='#')

# Ensure all 9 categories per image
rows = []
images = df['filename'].unique()
for img in images:
    for cat in CATEGORIES:
        count = grouped[(grouped['filename'] == img) & (grouped['Classifications'] == cat)]['#']
        rows.append({
            'img name': img,
            'Classifications': cat,
            '#': int(count.values[0]) if not count.empty else 0
        })

report_df = pd.DataFrame(rows)
report_df.to_excel("misclassification_report.xlsx", index=False)
print("✅ Excel file generated: misclassification_report.xlsx")
