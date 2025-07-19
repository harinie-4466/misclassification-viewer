import os
import xml.etree.ElementTree as ET

def extract_points_from_xml(xml_folder):
    """
    Extract level-800 training points from XML and categorize them by label.
    Returns a list of dictionaries for each coordinate.
    """
    all_points = []

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(path)
        root = tree.getroot()

        filename = root.findtext("filename")

        for obj in root.findall("object"):
            label = obj.findtext("name")
            if "800" not in label:
                continue

            coord_text = obj.findtext("coordinates")
            if not coord_text:
                continue

            coords = coord_text.strip().split(';')
            for coord in coords:
                try:
                    x, y = map(float, coord.split(','))
                    if label.startswith("green"):
                        category = "GREEN_800"
                    elif label.startswith("non-green"):
                        category = "NON_GREEN_800"
                    elif label.startswith("Coconut"):
                        category = "COCONUT_800"
                    else:
                        continue

                    all_points.append({
                        'filename': filename,
                        'x': x,
                        'y': y,
                        'category': category,
                        'rgb': ''
                    })
                except:
                    continue

    return all_points
