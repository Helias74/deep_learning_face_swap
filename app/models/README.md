**Les modèles ne sont pas versionnés dans Git** 
## Installation des modèles

### 1. Modèles de crop

```bash
cd app/models/crop

# YOLOv8n (6 MB)
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Modèle custom (si disponible)
# Copiez model.pth ici
```

### 2. Modèles de face swap

```bash
cd app/models/face_swap

# Buffalo L (280 MB)
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip -d buffalo_l
rm buffalo_l.zip

# Buffalo SC (100 MB)
curl -L -o buffalo_sc.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
unzip buffalo_sc.zip -d buffalo_sc
rm buffalo_sc.zip


app/models/
├── crop/
│   ├── yolov8n.pt
│   └── model.pth
└── face_swap/
├── buffalo_l/
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   ├── det_10g.onnx
│   ├── genderage.onnx
│   └── w600k_r50.onnx
└── buffalo_sc/
├── det_500m.onnx
├── det_2.5g.onnx
└── w600k_mbf.onnx

```
