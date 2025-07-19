from flask import Flask, render_template, jsonify
import pandas as pd
from utils import extract_points_from_xml

app = Flask(__name__)

PREDICTION_CSV = 'point_predictions.csv'
XML_FOLDER = 'annotations'

df_predictions = pd.read_csv(PREDICTION_CSV)

@app.route('/')
def index():
    images = df_predictions['filename'].unique().tolist()
    return render_template('index.html', images=images)

@app.route('/view/<filename>')
def view_image(filename):
    return render_template('view_image.html', filename=filename)

@app.route('/data/<filename>')
def get_data(filename):
    points = []

    subset = df_predictions[(df_predictions['filename'] == filename) & (df_predictions['result'] == 'wrong')]
    for _, row in subset.iterrows():
        category = get_misclassification_code(row['expected'], row['predicted'])
        points.append({
            'x': int(row['x']),
            'y': int(row['y']),
            'category': category,
            'rgb': row.get('rgb', '')
        })

    training_points = extract_points_from_xml(XML_FOLDER)
    for point in training_points:
        if point['filename'] == filename:
            points.append(point)

    return jsonify(points)

def get_misclassification_code(expected, predicted):
    if expected.startswith("Coconut") and predicted.startswith("green"):
        return "CG"
    elif expected.startswith("Coconut") and predicted.startswith("non"):
        return "CN"
    elif expected.startswith("non") and predicted.startswith("green"):
        return "NG"
    elif expected.startswith("green") and predicted.startswith("non"):
        return "GN"
    elif expected.startswith("non") and predicted.startswith("Coconut"):
        return "NC"
    elif expected.startswith("green") and predicted.startswith("Coconut"):
        return "GC"
    else:
        return "UNK"

if __name__ == '__main__':
    app.run(debug=True)
