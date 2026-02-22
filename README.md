# Cocoa Pod Disease: Hybrid CNN–ViT Classification + YOLOv8 Lesion Severity

This project implements the **two-stage pipeline** described in the research proposal:

1) **Classification** (Healthy / Black Pod Rot (BPR) / Frosty Pod Rot (FPR)) using:
   - CNN backbone (default: ResNet50)
   - ViT backbone (default: ViT-Base/16)
   - Fusion variants: CNN-only, ViT-only, Concat, **Attention Fusion** (+ optional EMA)

2) **Lesion localization** using **YOLOv8**, then compute a **severity score**:
   - `severity = lesion_area / pod_area`
   - Works best if your YOLO dataset includes two classes: `pod` and `lesion`.
   - If you only label `lesion`, severity can be computed relative to full image area as a fallback.

---

## 1) Run in Google Colab (recommended)

### A) Upload and unzip
Upload `cocoa_hybrid_project.zip` to Colab, then:

```bash
!unzip -q cocoa_hybrid_project.zip -d /content/cocoa_hybrid_project
%cd /content/cocoa_hybrid_project
```

### B) Install dependencies
```bash
!pip -q install -r requirements.txt
```

### C) Train classification
Prepare a classification dataset like:

```
data_cls/
  train/
    Healthy/
    BPR/
    FPR/
  val/
    Healthy/
    BPR/
    FPR/
  test/
    Healthy/
    BPR/
    FPR/
```

Then run:

```bash
!python -m src.train_cls --data_dir data_cls --variant attn --epochs 20 --batch_size 32
```

Variants: `cnn`, `vit`, `concat`, `attn`

### D) Train YOLOv8 detection
Prepare a YOLO dataset like:

```
data_det/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Example `data.yaml`:

```yaml
path: /content/cocoa_hybrid_project/data_det
train: images/train
val: images/val
names: [pod, lesion]
```

Train YOLO:

```bash
!python -m src.train_yolo --data_yaml data_det/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

### E) Inference + severity
```bash
!python -m src.infer_cls_and_severity \
  --cls_ckpt runs_cls/best.pt \
  --yolo_ckpt runs/detect/train/weights/best.pt \
  --image_path path/to/image.jpg
```

---

## 2) Notes
- This repo is **dataset-agnostic**: you must supply your own images/labels.
- For best severity estimation, label **pod** and **lesion** boxes.
- This code uses PyTorch + timm (for ResNet/ViT) + ultralytics (YOLOv8).

