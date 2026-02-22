# cocoa-test
## Cocoa Pod Disease Classification (V1–V3) + Optional YOLO Severity (%)

This repository supports **PhD-style experiments** for cocoa pod disease recognition using three comparable classification variants:

- **V1 — CNN Only**
- **V2 — ViT Only**
- **V3 — CNN + ViT (Feature Concatenation)**

An optional second stage (YOLO) can be used for **lesion localization** and **severity scoring (%)**.

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

Metrics reported:
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
- If only `lesion` is labeled, severity can only be estimated relative to image area (less reliable)

---

## 2) Preferred Dataset Folder (Classification)

✅ Preferred folder name: `data_cls/` (PyTorch `ImageFolder` format)

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

Keep the class folder names exactly: Healthy, BPR, FPR

If your original dataset uses Sana, Fito, Monilia, do mapping/renaming during preprocessing

3) Run in Google Colab (Recommended)
A) Get the code (recommended: clone from GitHub)
!git clone https://github.com/princesss96/cocoa-test.git
%cd cocoa-test

(Alternative: upload cocoa_hybrid_project.zip and unzip)

!unzip -q cocoa_hybrid_project.zip -d /content/cocoa_hybrid_project
%cd /content/cocoa_hybrid_project
B) Install dependencies
!pip -q install -r requirements.txt
C) Point to your dataset (Google Drive)
from google.colab import drive
drive.mount("/content/drive")

Example path:

/content/drive/MyDrive/data_cls
D) Train V1 / V2 / V3 (choose one)

✅ Tip: use a unique --out_dir per run to avoid overwriting results
✅ Tip: use a fixed --seed for reproducibility

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
E) Save training output to a log file (optional but recommended)
!mkdir -p logs

!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls \
  --variant vit \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir runs_cls_v2_vit 2>&1 | tee logs/v2_vit_train.log

Outputs:

Checkpoints + metrics are saved into the folder you set in --out_dir (e.g., runs_cls_v2_vit/)

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
6) Reproducibility Tips (Important for PhD Experiments)

To compare V1 vs V2 vs V3 fairly:

Use the same dataset split (train/val/test)

Keep these identical across variants:

--epochs, --batch_size, --img_size, --lr, --weight_decay

augmentation settings (do not change between variants)

Use the same seed (e.g., --seed 42)

Use separate --out_dir per run

7) Common Colab Mistakes

If you see SyntaxError: invalid syntax when running shell commands, you probably ran it as Python.

Use:

!command ... (recommended)

or %%bash cell magic

Example:

!python -m src.train_cls --data_dir data_cls --variant cnn
Publish README changes to GitHub (Mac)
cd ~/Projects/cocoa-test
open -e README.md   # paste this README, save, close

git add README.md
git commit -m "Update README formatting and V1–V3 Colab instructions"
git pull --rebase origin main
git push
::contentReference[oaicite:0]{index=0}