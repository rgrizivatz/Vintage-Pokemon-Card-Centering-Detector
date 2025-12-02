# Pokémon Card Yellow Frame Border Analyzer (Beta)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-BETA-orange.svg)]()
[![License](https://img.shields.io/badge/License-Custom-green.svg)]()

## ⚠️ Beta Version Notice
This project is currently in **BETA**.  
The detection algorithm is calibrated specifically for:
- Japanese vintage Pokémon cards  
- With the old blue back  
- And the classic thick yellow border  

Support for other card types is experimental.

---

## 🎯 What This Tool Does
This tool analyzes Pokémon card centering by extracting:
- Border thickness (left/right/top/bottom)
- Quartiles (Q1, median, Q3)
- PSA-style centering (Left–Right, Top–Bottom)

It also generates:
- Red border overlay  
- Clean segment lines  
- Diagnostic histograms  
- Multi-panel visualization  

---

## ⭐ Current Accuracy & Limitations

### ✔ Works excellently for:
- Vintage Japanese Pokémon cards  
- Yellow-framed cards  
- Non-yellow artwork backgrounds  

### ✔ Works *pretty well* for:
- Most Pokémon types  
- Many electric-type Pokémon  

### ⚠ Difficult cases:
- Electric Pokémon with strong yellow backgrounds  
- Full yellow artwork backgrounds  

### ❌ Not supported yet:
- Modern borderless cards  
- Silver-bordered 2023+ cards  

---

# 🧩 Software Architecture

## High-Level System Overview

The project follows a **pipeline architecture**:

```
Raw Image
    ↓
Preprocessing (grayscale, blur, edges)
    ↓
Card Silhouette Detection (contours)
    ↓
Yellow Border Isolation (HSV mask + distance transform)
    ↓
Border Thickness Extraction (left/right/top/bottom)
    ↓
Outlier Cleaning (IQR mask)
    ↓
Quartile Measurements + PSA Centering
    ↓
Visualization Rendering
```

---

## Detailed Block Diagram (ASCII)

```
                    +------------------------+
                    | yellow_frame_detector  |
                    |         .py            |
                    +-----------+------------+
                                |
                                v
                       +----------------+
                       | Load config    |
                       +----------------+
                                |
                                v
                     +----------------------+
                     | Load + preprocess    |
                     | image (PIL → NumPy)  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Card silhouette      |
                     | detection (contours) |
                     +----------+-----------+
                                |
                                v
                +----------------------------------+
                | Yellow border mask detection     |
                | (HSV threshold + distance map)   |
                +----------+-----------------------+
                           |
                           v
               +------------------------------+
               | Thickness profiles & segments|
               +---------------+--------------+
                               |
                               v
                 +-----------------------------+
                 | Outlier cleaning (IQR)      |
                 +---------------+-------------+
                               |
                               v
           +------------------------------------------+
           | Statistics: quartiles + centering        |
           +--------------------+---------------------+
                               |
                               v
                     +------------------------+
                     | Visualization & output |
                     +------------------------+
```

---

## 📁 Project Structure

```
pokemon-border-analyzer/
├── yellow_frame_detector.py
├── config.yaml
├── yellow_frame_detector_example.ipynb
├── logo.svg
├── LICENSE.txt
└── README.md
```

---

## 🚀 Usage

### CLI

```bash
python yellow_frame_detector.py mycard.webp --config config.yaml
```

### Python

```python
from yellow_frame_detector import run_pipeline
result = run_pipeline("mycard.webp", "config.yaml")
```

### Notebook

Open `yellow_frame_detector_example.ipynb`.

---

## 🔮 Roadmap
- Improved detection for electric Pokémon  
- Complete support for all Pokémon card types  
- LAB color clustering  
- ML-based PSA-grade prediction  

---

## © License
Copyright © 2025 Richard Grizivatz  
Pokémon Card Yellow Frame Border Analyzer  
All rights reserved.

See `LICENSE.txt` for full terms.
