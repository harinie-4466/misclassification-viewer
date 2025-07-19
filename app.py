import os
import cv2
import pandas as pd
from flask import Flask, jsonify, render_template_string, send_from_directory, request

app = Flask(__name__)

# Your existing LABEL_MAP
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

def get_rgb_at_point(image_path, x, y):
    """Get RGB values at a specific point in the image"""
    img = cv2.imread(image_path)
    if img is None or y >= img.shape[0] or x >= img.shape[1] or x < 0 or y < 0:
        return None
    
    # OpenCV uses BGR, convert to RGB
    bgr = img[y, x]
    rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return rgb

def map_label_to_category(predicted_label):
    """Map predicted labels to display categories"""
    label_to_category = {
        'green-800': 'CG',        # Correct Green
        'non-green-800': 'CN',    # Correct Non-green  
        'Coconut-800': 'CC',      # Correct Coconut
        'green-400': 'CG',
        'non-green-400': 'CN',
        'green-300': 'CG', 
        'non-green-300': 'CN',
        'Coconut-300': 'CC',
        'Cocount-400': 'CC',      # Note: this seems to be a typo in your original code
    }
    return label_to_category.get(predicted_label, 'NC')  # Default to NC (Not Classified)

@app.route('/')
def index():
    """List all images"""
    try:
        images_dir = 'images'
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>Image Viewer</title></head>
            <body>
                <h1>Available Images</h1>
                <ul>
                {% for image in images %}
                    <li><a href="/view/{{ image }}">{{ image }}</a></li>
                {% endfor %}
                </ul>
            </body>
            </html>
            """
            return render_template_string(html, images=images)
        else:
            return "Images directory not found"
    except Exception as e:
        return f"Error: {e}"

@app.route('/view/<filename>')
def view_image(filename):
    """Render the image viewer page"""
    # The HTML template (your fixed view_image.html content)
    html_template = """<!DOCTYPE html>
<html>
<head>
  <title>Image Viewer - {{ filename }}</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: black;
      color: white;
    }

    .image-container {
      width: 100vw;
      height: 100vh;
      position: relative;
      overflow: hidden;
    }

    #mainImage, #overlayCanvas {
      position: absolute;
      top: 0;
      left: 0;
      transform-origin: top left;
    }

    #zoomControls {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1001;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .zoom-group {
      display: flex;
      gap: 6px;
    }

    .zoom-button {
      font-size: 22px;
      padding: 4px 12px;
      background: white;
      color: black;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }

    .floating-toggles {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) scale(1);
      background: rgba(0, 0, 0, 0.3);
      padding: 20px;
      border-radius: 15px;
      z-index: 1000;
      transition: transform 0.3s;
    }

    .toggle {
      font-size: 18px;
      display: flex;
      align-items: center;
      color: white;
      margin: 6px 0;
    }

    .toggle input[type="checkbox"] {
      width: 18px;
      height: 18px;
      margin-right: 10px;
    }

    .color-box {
      width: 30px;
      height: 30px;
      text-align: center;
      line-height: 30px;
      font-size: 14px;
      font-weight: bold;
      color: black;
      border: 1px solid #aaa;
      margin-right: 10px;
    }

    #tooltip {
      position: absolute;
      background: white;
      color: black;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 14px;
      display: none;
      pointer-events: none;
      z-index: 1000;
    }

    .training-points-section {
      border-top: 1px solid #666;
      margin-top: 15px;
      padding-top: 15px;
    }

    .level-toggle {
      display: flex;
      align-items: center;
      margin: 8px 0;
      font-size: 16px;
    }

    .level-toggle input[type="checkbox"] {
      width: 16px;
      height: 16px;
      margin-right: 8px;
    }

    .level-label {
      font-weight: bold;
      color: #ffff99;
    }
  </style>
</head>
<body>
  <div id="zoomControls">
    <div class="zoom-group">
      <button class="zoom-button" onclick="adjustZoom(-0.1)">− Image</button>
      <button class="zoom-button" onclick="adjustZoom(0.1)">+ Image</button>
    </div>
    <div class="zoom-group">
      <button class="zoom-button" onclick="adjustToggleZoom(-0.1)">− Panel</button>
      <button class="zoom-button" onclick="adjustToggleZoom(0.1)">+ Panel</button>
    </div>
  </div>

  <div class="image-container">
    <img id="mainImage" src="/static/images/{{ filename }}" />
    <canvas id="overlayCanvas"></canvas>

    <div id="togglePanel" class="floating-toggles">
      <label>Dot Size:
        <input type="number" id="dotSizeInput" value="10" min="1" max="100" />
      </label>
      <div id="checkboxes"></div>
      
      <div class="training-points-section">
        <div class="level-label">Training Data Points (Level 800):</div>
        <label class="level-toggle">
          <input type="checkbox" id="showTraining800" checked> Show Training Points
        </label>
      </div>
    </div>

    <div id="tooltip"></div>
  </div>

  <script>
    let zoomScale = 1;
    let toggleScale = 1;

    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('mainImage');
    const tooltip = document.getElementById('tooltip');
    let points = [];
    let trainingPoints = [];

    const categoryColors = {
      CG: 'red', CN: 'blue', NG: 'green',
      GN: 'orange', NC: 'purple', CC: 'cyan'
    };

    const trainingColors = {
      'green-800': '#00ff00',
      'non-green-800': '#ff0000',
      'Coconut-800': '#8B4513'
    };

    function resizeCanvas() {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }

    function applyZoom() {
      img.style.transform = `scale(${zoomScale})`;
      canvas.style.transform = `scale(${zoomScale})`;
      draw();
    }

    function adjustZoom(delta) {
      zoomScale = Math.max(0.1, zoomScale + delta);
      applyZoom();
    }

    function adjustToggleZoom(delta) {
      toggleScale = Math.max(0.5, toggleScale + delta);
      const togglePanel = document.getElementById('togglePanel');
      togglePanel.style.transform = `translate(-50%, -50%) scale(${toggleScale})`;
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const userSize = parseInt(document.getElementById("dotSizeInput").value) || 10;
      const radius = Math.min(userSize, 100);
      const selected = getSelectedCategories();

      points.forEach(p => {
        if (p.x < 0 || p.y < 0 || p.x > canvas.width || p.y > canvas.height) return;
        
        if (selected.includes(p.category)) {
          ctx.fillStyle = categoryColors[p.category] || 'gray';
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
          ctx.fill();
        }
      });

      const showTraining = document.getElementById("showTraining800").checked;

      if (showTraining) {
        trainingPoints.forEach(tp => {
          if (tp.x < 0 || tp.y < 0 || tp.x > canvas.width || tp.y > canvas.height) return;
          
          if (tp.level === 800) {
            ctx.fillStyle = trainingColors[tp.label] || 'white';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(tp.x, tp.y, 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
          }
        });
      }
    }

    function getSelectedCategories() {
      return Array.from(document.querySelectorAll('#checkboxes input:checked')).map(cb => cb.value);
    }

    function renderToggles(countMap) {
      const box = document.getElementById('checkboxes');
      box.innerHTML = '';
      for (const [cat, color] of Object.entries(categoryColors)) {
        const label = document.createElement('label');
        label.className = 'toggle';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = cat;

        const colorBox = document.createElement('span');
        colorBox.className = 'color-box';
        colorBox.style.backgroundColor = color;
        colorBox.textContent = countMap[cat] || 0;

        label.appendChild(input);
        label.appendChild(colorBox);
        label.append(` ${cat}`);
        box.appendChild(label);
      }

      document.querySelectorAll('#checkboxes input').forEach(cb => {
        cb.addEventListener('change', () => draw());
      });
    }

    function loadTrainingData(filename) {
      fetch(`/training_data/${filename}`)
        .then(res => res.json())
        .then(data => {
          trainingPoints = data;
          draw();
        })
        .catch(err => {
          console.log("Training data not available:", err);
        });
    }

    img.onload = () => {
      resizeCanvas();
      
      fetch(`/data/${filename}`).then(res => res.json()).then(data => {
        points = data.map(p => ({
          ...p,
          rgb: p.rgb || "(?, ?, ?)"
        }));

        const countMap = {};
        for (const p of points) {
          countMap[p.category] = (countMap[p.category] || 0) + 1;
        }

        renderToggles(countMap);
        applyZoom();
      });

      loadTrainingData(filename);
    };

    document.getElementById("dotSizeInput").addEventListener('input', draw);
    document.getElementById("showTraining800").addEventListener('change', draw);

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX / zoomScale;
      const y = (e.clientY - rect.top) * scaleY / zoomScale;

      let hoveredPoint = null;
      for (const tp of trainingPoints) {
        const dist = Math.sqrt((x - tp.x) ** 2 + (y - tp.y) ** 2);
        if (dist <= 6) {
          hoveredPoint = tp;
          break;
        }
      }

      if (hoveredPoint) {
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.clientX + 10}px`;
        tooltip.style.top = `${e.clientY - 30}px`;
        tooltip.innerHTML = `Training: ${hoveredPoint.label}<br>Level: ${hoveredPoint.level}<br>RGB: ${hoveredPoint.rgb || 'N/A'}`;
      } else {
        tooltip.style.display = 'none';
      }
    });

    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });

    // Replace filename in the script
    const filename = "{{ filename }}";
  </script>
</body>
</html>"""
    
    return render_template_string(html_template, filename=filename)

@app.route('/static/images/<filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    return send_from_directory('images', filename)

@app.route('/data/<filename>')
def get_prediction_data(filename):
    """Get prediction/category data for display"""
    try:
        # Load prediction results if they exist
        prediction_file = 'point_predictions.csv'
        if os.path.exists(prediction_file):
            df = pd.read_csv(prediction_file)
            image_data = df[df['filename'] == filename]
            
            points = []
            for _, row in image_data.iterrows():
                category = map_label_to_category(row['predicted'])
                
                points.append({
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'category': category,
                    'rgb': f"Predicted: {row['predicted']}"
                })
            
            return jsonify(points)
        else:
            # Return sample data if no predictions file exists
            return jsonify([
                {'x': 100, 'y': 100, 'category': 'CG', 'rgb': '(255,0,0)'},
                {'x': 200, 'y': 200, 'category': 'CN', 'rgb': '(0,255,0)'},
                {'x': 300, 'y': 300, 'category': 'CC', 'rgb': '(0,0,255)'}
            ])
    
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return jsonify([])

@app.route('/training_data/<filename>')
def get_training_data(filename):
    """Endpoint to get training data points for a specific image (level 800 only)"""
    try:
        # Load the CSV with annotations
        df = pd.read_csv('all_annotations.csv')
        
        # Filter for the specific filename and level 800 labels only
        image_data = df[df['filename'] == filename]
        level_800_labels = ['green-800', 'non-green-800', 'Coconut-800']
        image_data = image_data[image_data['label'].isin(level_800_labels)]
        
        training_points = []
        
        for _, row in image_data.iterrows():
            label = row['label']
            
            # Parse the points
            point_strs = row['points'].split(';')
            image_path = os.path.join('images', filename)
            
            for point_str in point_strs:
                try:
                    x, y = map(float, point_str.strip().split(','))
                    x, y = int(x), int(y)
                    
                    # Get RGB values at this point
                    rgb = get_rgb_at_point(image_path, x, y)
                    rgb_str = f"({rgb[0]}, {rgb[1]}, {rgb[2]})" if rgb else "N/A"
                    
                    training_points.append({
                        'x': x,
                        'y': y,
                        'label': label,
                        'level': 800,
                        'rgb': rgb_str
                    })
                except (ValueError, IndexError):
                    continue
        
        return jsonify(training_points)
    
    except Exception as e:
        print(f"Error loading training data: {e}")
        return jsonify([])

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Make sure you have these files in the same directory:")
    print("- all_annotations.csv")
    print("- images/ folder with your images")
    print("- point_predictions.csv (optional)")
    print("\nServer will start at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)










'''from flask import Flask, render_template, jsonify
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
'''