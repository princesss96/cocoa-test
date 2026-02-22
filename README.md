# cocoa-test
## Cocoa Pod Disease Classification (V1–V3) + Optional YOLOv8 Severity (%)

cocoa pod disease recognition using three comparable classification variants:

- **V1 — CNN Only**
- **V2 — ViT Only**
- **V3 — CNN + ViT (Feature Concatenation)**

(Optional) A second stage (**YOLOv8**) can be used for **lesion localization** and **severity scoring (%)**.

---

## 1) Project Overview

### 1.1 Classification Task (Image-level)
Classes:
- **Healthy**
- **BPR** (Black Pod Rot)
- **FPR** (Frosty Pod Rot)

Variants:
- **V1 (CNN Only)**: CNN backbone (e.g., ResNet50)
- **V2 (ViT Only)**: ViT backbone (e.g., ViT-Base/16)
- **V3 (CNN+ViT Concat)**: concatenated features from CNN + ViT

Metrics (typical):
- Accuracy
- Macro-F1
- Precision (Macro)
- Recall (Macro)
- Cohen’s Kappa

### 1.2 Optional Severity (Percentage)
Severity is computed as:

**Severity (%) = (Total Lesion Area / Pod Area) × 100**

Best practice:
- YOLO dataset should include two classes: `pod` and `lesion`
- If only `lesion` is labeled, severity is estimated relative to image area (less reliable)

---

## 2) Preferred Dataset Folder (Classification)

✅ Preferred folder name: `data_cls/` (PyTorch ImageFolder format)

Expected layout:

```text
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

Notes:

Keep class folder names exactly: Healthy, BPR, FPR

If your original dataset uses other names (e.g., Sana/Fito/Monilia), do renaming/mapping during preprocessing.

3) Run in Google Colab (Recommended)
A) Get the code (recommended: clone from GitHub)
!git clone https://github.com/princesss96/cocoa-test.git
%cd cocoa-test
B) Install dependencies
!pip -q install -r requirements.txt
C) Mount Drive (if dataset is in Google Drive)
from google.colab import drive
drive.mount("/content/drive")

Example dataset path:

/content/drive/MyDrive/data_cls

D) Train V1 / V2 / V3

✅ Tip: use a unique --out_dir per run to avoid overwriting.
✅ Tip: keep --seed fixed for fair comparison.

V1 — CNN Only

!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls \
  --variant cnn \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir runs_cls_v1_cnn

V2 — ViT Only

!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls \
  --variant vit \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir runs_cls_v2_vit

V3 — CNN + ViT Concat

!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls \
  --variant concat \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir runs_cls_v3_concat
E) Save training output to a log file (recommended)
!mkdir -p logs

!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls \
  --variant vit \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir runs_cls_v2_vit 2>&1 | tee logs/v2_vit_train.log

Outputs:

Checkpoints + metrics are saved into the folder you set in --out_dir.

4) Optional: YOLOv8 Lesion Detection + Severity (%)
4.1 YOLO Dataset Layout (example)
data_det/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml

Example data.yaml:

path: /content/cocoa-test/data_det
train: images/train
val: images/val
names: [pod, lesion]

Train YOLO:

!python -m src.train_yolo \
  --data_yaml data_det/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
5) Inference (Classification + Optional Severity)
!python -m src.infer_cls_and_severity \
  --cls_ckpt runs_cls_v3_concat/best.pt \
  --yolo_ckpt runs/detect/train/weights/best.pt \
  --image_path path/to/image.jpg
6) Reproducibility Tips

To compare V1 vs V2 vs V3 fairly:

Use the same dataset split (train/val/test)

Keep these fixed across variants:

--epochs, --batch_size, --img_size, --lr, --weight_decay

Keep augmentation settings the same

Use the same seed (e.g., --seed 42)

Use separate --out_dir per run
