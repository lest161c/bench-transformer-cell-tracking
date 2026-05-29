import xml.etree.ElementTree as ET
import re
import sys

def parse_val(text):
    if text.endswith('k'):
        return float(text[:-1]) * 1000
    return float(text)

def main():
    svg_file = '/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/eval_tensorboard/val_loss_22_04_26.svg'
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    x_map = []
    y_map = []
    
    for t in root.iter('{http://www.w3.org/2000/svg}text'):
        text = t.text
        x_pix = float(t.attrib.get('x', 0))
        y_pix = float(t.attrib.get('y', 0))
        
        try:
            val = parse_val(text)
        except ValueError:
            continue
            
        if 'k' in text or (y_pix == 0 and x_pix != 0):
            x_map.append((x_pix, val))
        else:
            y_map.append((y_pix, val))
            
    # Add exception for -50k which has x=0, y=0
    x_map.append((0, -50000))
    y_map.append((0, 0.35))
    
    # Calculate mapping (slope and intercept)
    # data_x = m_x * pix_x + b_x
    if len(x_map) >= 2:
        m_x = (x_map[1][1] - x_map[0][1]) / (x_map[1][0] - x_map[0][0])
        b_x = x_map[0][1] - m_x * x_map[0][0]
    else:
        m_x, b_x = 1, 0
        
    if len(y_map) >= 2:
        # Sort y_map by pix_y to ensure distinct points
        y_map = sorted(list(set(y_map)))
        m_y = (y_map[1][1] - y_map[0][1]) / (y_map[1][0] - y_map[0][0])
        b_y = y_map[0][1] - m_y * y_map[0][0]
    else:
        m_y, b_y = 1, 0
        
    def pix_to_data(px, py):
        return m_x * px + b_x, m_y * py + b_y

    curves = {}
    
    for p in root.iter('{http://www.w3.org/2000/svg}path'):
        d = p.attrib.get('d', '')
        stroke = p.attrib.get('stroke', '')
        
        if not d or not stroke or len(d) < 20:
            continue
            
        # extract coords
        points_str = re.findall(r'[ML]([0-9.-]+,[0-9.-]+)', d)
        points = []
        for pt in points_str:
            px, py = map(float, pt.split(','))
            points.append(pix_to_data(px, py))
            
        if len(points) < 5:
            continue
            
        if stroke not in curves or len(points) > len(curves[stroke]):
            curves[stroke] = points
            
    print("==================================================")
    print(" Validation Loss Curves Summary")
    print("==================================================")
    
    for color, points in curves.items():
        points.sort(key=lambda pt: pt[0])
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        
        min_y = min(pt[1] for pt in points)
        
        # Convergence step: 95% of final loss drop
        target_y = start_y - 0.95 * (start_y - min_y)
        conv_step = end_x
        for x, y in points:
            if y <= target_y:
                conv_step = x
                break
                
        print(f"Curve Color : {color}")
        print(f"  Start Loss: {start_y:.4f} (at step {start_x:.0f})")
        print(f"  End Loss  : {end_y:.4f} (at step {end_x:.0f})")
        print(f"  Min Loss  : {min_y:.4f}")
        print(f"  Speed     : reaches 95% of its total loss drop at step {conv_step:.0f}")
        print("--------------------------------------------------")

if __name__ == '__main__':
    main()
