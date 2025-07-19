import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Define label mappings
LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "green-400": 2,
    "non-green-400": 3,
    "green-300": 4,
    "non-green-300": 5,
    "Coconut-300": 6,
    "Cocount-400": 7,
    "Coconut-800": 8
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Paths
MODEL_PATH = 'model/point_classifier.h5'
CSV_PATH = 'all_annotations.csv'
IMAGE_FOLDER = 'images'
OUTPUT_CSV = 'point_predictions.csv'  # <-- ✅ Saved here

def evaluate_point_predictions():
    # Load model
    model = load_model(MODEL_PATH)

    # Load point data
    df = pd.read_csv(CSV_PATH)
    results = []
    correct, total = 0, 0

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        expected_label = LABEL_MAP.get(row['label'], -1)
        if expected_label == -1:
            continue

        point_strs = row['points'].split(';')
        for point_str in point_strs:
            x, y = map(float, point_str.strip().split(','))
            x, y = int(x), int(y)

            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                rgb = img[y, x] / 255.0
                rgb = rgb.reshape(1, -1)
                pred = np.argmax(model.predict(rgb, verbose=0))
                is_correct = pred == expected_label
                total += 1
                correct += int(is_correct)
                results.append({
                    'filename': row['filename'],
                    'x': x,
                    'y': y,
                    'expected': INV_LABEL_MAP[expected_label],
                    'predicted': INV_LABEL_MAP[pred],
                    'result': 'correct' if is_correct else 'wrong'
                })

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved predictions to {OUTPUT_CSV}")
    print(f"🎯 Accuracy: {correct}/{total} = {correct / total:.2%}" if total else "No predictions made")

if __name__ == '__main__':
    evaluate_point_predictions()

