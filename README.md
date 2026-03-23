# cocoa-test — Cocoa Pod Disease (Classification + YOLOv8 Pod–Lesion Localization for Severity Estimation)

Repo ini mengandungi eksperimen untuk:

1. **Image-level disease classification** menggunakan CNN / ViT / Fusion (Concat & Proposed Attention Fusion + EMA)
2. **Pod + lesion localization** menggunakan **YOLOv8** (bounding box detection) untuk menyokong **severity estimation**
3. Pengiraan **severity (%)** berdasarkan nisbah kawasan lesion kepada kawasan pod
4. Semua output/hasil eksperimen disimpan di Google Drive: `cocoa_runs`

---

## Google Drive Paths (Reference)

**Dataset asal (raw)**
- `/content/drive/MyDrive/data_cocoa_original`

**Output eksperimen (logs, checkpoints, runs)**
- `/content/drive/MyDrive/cocoa_runs`

**Classification dataset (grouped split, leak-free)**
- `/content/drive/MyDrive/data_cls_crop_grouped`

**YOLO-ready dataset (pod + lesion)**
- `/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1`

---

## Classes / Labels

### Classification (3 kelas)
- `Fitoftora`
- `Monilia`
- `Sana`

> Thesis reporting label mapping:
> - `Fitoftora` → `Black Pod Rot (BPR)`
> - `Monilia` → `Frosty Pod Rot (FPR)`
> - `Sana` → `Healthy`

### Detection (YOLO, 2 classes: pod + lesion)

Mapping class id (YOLO txt):
- `0: pod`
- `1: lesion`

YOLO label format (normalized):
```text
class x_center y_center width height

Note:
Disease category prediction is handled by the classification branch.
The YOLO branch is used only for object localization of the cocoa pod and visible lesion regions.
Severity is then computed as:

Severity (%) = (total lesion area inside pod / pod area) × 100

Repo Structure
cocoa-test/
├── src/
│   ├── train_cls.py              # classification training entry
│   ├── data.py                   # ImageFolder loaders + transforms
│   ├── models.py                 # HybridClassifier variants: cnn/vit/concat/attn
│   ├── ema.py                    # EMA helper
│   ├── utils.py                  # metrics + checkpoint helpers
│   └── train_v4_stack.py         # optional: CNN-features + PCA + stacking baseline
│
├── scripts/
│   ├── build_yolo_pod_lesion_dataset.py   # build pod+lesion YOLO dataset
│   ├── severity_from_labels.py            # compute GT severity from YOLO labels
│   ├── estimate_severity_from_yolo.py     # compute predicted severity from YOLO detections
│   └── compare_severity_csv.py            # compare GT vs predicted severity
A) Colab Setup (Start from Scratch)
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf cocoa-test
!git clone https://github.com/princesss96/cocoa-test.git
%cd /content/cocoa-test

!pip -q install -U timm scikit-learn ultralytics

!mkdir -p logs
!mkdir -p /content/drive/MyDrive/cocoa_runs/logs
B) Classification (ImageFolder)
Dataset Structure (Classification)

Expected structure:

data_cls/
  train/Fitoftora/*.jpg
  train/Monilia/*.jpg
  train/Sana/*.jpg
  val/...
  test/...

My runs use (grouped split, leak-free):

/content/drive/MyDrive/data_cls_crop_grouped
Train Command Template
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant <cnn|vit|concat|attn> \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/<RUN_NAME> \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/<LOG_NAME>.log
Variants (Thesis IDs)
V1 = CNN baseline (--variant cnn)
V2 = ViT baseline (--variant vit)
V5 = CNN+ViT Concat baseline (--variant concat)
V6 = Proposed Attention Fusion (--variant attn)
V6b = Proposed Attention Fusion + EMA tuned (--variant attn --use_ema --ema_decay 0.99)

Negative ablation:
EMA yang terlalu tinggi seperti ema_decay=0.999 boleh collapse kepada kelas majoriti.

Run Commands (Copy-Paste)
V1 — CNN
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant cnn \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/runs_cls_v1_cnn \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/v1_cnn.log
V2 — ViT
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant vit \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/runs_cls_v2_vit \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/v2_vit.log
V5 — CNN + ViT Concat
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant concat \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/runs_cls_v5_concat_grouped \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/v5_concat_grouped.log
V6 — Proposed Attention Fusion (Best classification so far)
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant attn \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/runs_cls_v6_attn_grouped \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/v6_attn_grouped.log
V6b — Proposed Attention Fusion + EMA (tuned)
!python -m src.train_cls \
  --data_dir /content/drive/MyDrive/data_cls_crop_grouped \
  --variant attn \
  --use_ema \
  --ema_decay 0.99 \
  --epochs 20 \
  --batch_size 32 \
  --seed 42 \
  --out_dir /content/drive/MyDrive/cocoa_runs/runs_cls_v6b_attn_ema_decay099 \
  2>&1 | tee /content/drive/MyDrive/cocoa_runs/logs/v6b_attn_ema_decay099.log
Output Files (Classification)

Each run folder saves:

best.pt → best macro-F1 on validation set
last.pt → last epoch checkpoint
console log in /content/drive/MyDrive/cocoa_runs/logs/*.log
Classification Results Summary (My latest)
ID	Model	Test Acc	Macro-F1	Kappa
V1	CNN	0.9137	0.7737	0.7231
V2	ViT	0.8849	0.7109	0.6032
V5	Concat	0.8705	0.6877	0.6115
V6	Attn Fusion (Proposed)	0.9281	0.8035	0.7565
V6b	Attn+EMA (0.99)	0.8993	0.7520	0.6770
C) YOLOv8 Pod–Lesion Localization (Detection for Severity Estimation)
1) Build YOLO-ready dataset for pod + lesion detection

This script:

reads raw images and corresponding pod + lesion YOLO annotation files
copies them into yolo_cocoa_pod_lesion_v1/images_all and yolo_cocoa_pod_lesion_v1/labels_all
creates a leak-free grouped split into images/{train,val,test} and labels/{train,val,test}
writes data.yaml with:
0: pod
1: lesion
import os, glob, shutil, random, re
from collections import defaultdict
from pathlib import Path

random.seed(42)

RAW = "/content/drive/MyDrive/data_cocoa_original"
OUT = "/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1"

img_all = Path(OUT) / "images_all"
lbl_all = Path(OUT) / "labels_all"
img_all.mkdir(parents=True, exist_ok=True)
lbl_all.mkdir(parents=True, exist_ok=True)

# (A) Copy all pairs with safe unique names: <Class>_<Stem>.jpg/.txt
imgs = glob.glob(os.path.join(RAW, "**", "*.jpg"), recursive=True) + \
       glob.glob(os.path.join(RAW, "**", "*.jpeg"), recursive=True)

copied = 0
for img in imgs:
    cls = os.path.basename(os.path.dirname(img))
    stem, ext = os.path.splitext(os.path.basename(img))
    lbl = os.path.splitext(img)[0] + ".txt"
    if not os.path.exists(lbl):
        continue

    new_stem = f"{cls}_{stem}"
    shutil.copy2(img, img_all / (new_stem + ext.lower()))
    shutil.copy2(lbl, lbl_all / (new_stem + ".txt"))
    copied += 1

print("Copied pairs:", copied)

# (B) Leak-free split by base id (handles *_boxN if exists)
def base_id(stem):
    m = re.match(r"(.+?)_box\d+$", stem)
    return m.group(1) if m else stem

pairs = sorted([p for p in img_all.glob("*.jpg")] + [p for p in img_all.glob("*.jpeg")])
groups = defaultdict(list)
for p in pairs:
    groups[base_id(p.stem)].append(p)

keys = list(groups.keys())
random.shuffle(keys)
n = len(keys)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_keys = set(keys[:n_train])
val_keys = set(keys[n_train:n_train+n_val])
test_keys = set(keys[n_train+n_val:])

for split in ["train", "val", "test"]:
    (Path(OUT) / "images" / split).mkdir(parents=True, exist_ok=True)
    (Path(OUT) / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_pair(img_path, split):
    stem = img_path.stem
    ext = img_path.suffix.lower()
    lbl_path = lbl_all / f"{stem}.txt"
    shutil.copy2(img_path, Path(OUT) / "images" / split / f"{stem}{ext}")
    shutil.copy2(lbl_path, Path(OUT) / "labels" / split / f"{stem}.txt")

for k, imgs_k in groups.items():
    split = "train" if k in train_keys else ("val" if k in val_keys else "test")
    for img in imgs_k:
        copy_pair(img, split)

print("Unique base images:", len(train_keys), len(val_keys), len(test_keys))
print("Overlap check:", len(train_keys & val_keys), len(train_keys & test_keys), len(val_keys & test_keys))

# (C) Write data.yaml
yaml_path = Path(OUT) / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {OUT}
train: images/train
val: images/val
test: images/test

names:
  0: pod
  1: lesion
""")

print("Wrote:", yaml_path)
2) Train YOLOv8 (baseline)
YOLOv8n (fast baseline)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    project="/content/drive/MyDrive/cocoa_runs/yolo",
    name="yolov8n_pod_lesion_v1",
    seed=42
)
YOLOv8n (longer run, higher imgsz)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1/data.yaml",
    epochs=200,
    imgsz=832,
    batch=16,
    patience=80,
    project="/content/drive/MyDrive/cocoa_runs/yolo",
    name="yolov8n_pod_lesion_p200",
    seed=42,
    cache=True
)
YOLOv8s (stronger detector)
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1/data.yaml",
    epochs=200,
    imgsz=832,
    batch=16,
    patience=80,
    close_mosaic=15,
    mixup=0.0,
    copy_paste=0.0,
    project="/content/drive/MyDrive/cocoa_runs/yolo",
    name="yolov8s_pod_lesion_img832",
    seed=42,
    cache=True
)
3) Evaluate YOLO on TEST
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/cocoa_runs/yolo/yolov8n_pod_lesion_p200/weights/best.pt")
metrics = model.val(
    data="/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1/data.yaml",
    split="test",
    imgsz=832,
    project="/content/drive/MyDrive/cocoa_runs/yolo",
    name="yolov8n_pod_lesion_p200_test",
    exist_ok=True
)
print(metrics.results_dict)

The YOLO metrics above evaluate detection quality only.
They do not directly measure severity estimation performance.

4) Resume Training (if training not finished)

If a run finished at epochs=200, you cannot resume unless you increase epochs.

from ultralytics import YOLO

last = "/content/drive/MyDrive/cocoa_runs/yolo/yolov8s_pod_lesion_img832/weights/last.pt"
model = YOLO(last)
model.train(resume=True, epochs=300)
5) Audit pod/lesion predictions / Error Analysis (save_txt)
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/cocoa_runs/yolo/yolov8n_pod_lesion_p200/weights/best.pt")

model.predict(
    source="/content/drive/MyDrive/yolo_cocoa_pod_lesion_v1/images/test",
    imgsz=832,
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    save_conf=True,
    project="/content/drive/MyDrive/cocoa_runs/audit",
    name="test_pred_conf025",
    exist_ok=True
)
Detection Results (Current Status)

Earlier YOLO runs in this repository were based on an older detection setup.
For proposal-aligned severity estimation, YOLO must be trained using pod + lesion annotations.
Updated pod–lesion detection results will be reported after retraining.

D) Severity Estimation

Severity is computed after pod and lesion localization.

Formula
Severity (%) = (total lesion area inside pod / pod area) × 100
Outputs
Ground-truth severity from annotation files
Predicted severity from YOLO detections
Comparison metrics:
MAE
RMSE
Correlation
Script Usage
python scripts/severity_from_labels.py
python scripts/estimate_severity_from_yolo.py
python scripts/compare_severity_csv.py
Suggested Evaluation Logic
severity_from_labels.py
computes annotation-derived ground-truth severity from YOLO label files
estimate_severity_from_yolo.py
computes predicted severity from detected pod and lesion boxes
compare_severity_csv.py
compares predicted severity vs ground truth using MAE, RMSE, and correlation
E) Notes for Thesis / Proposal Alignment
Classification branch predicts disease class:
Fitoftora / Monilia / Sana
reported in thesis as BPR / FPR / Healthy
YOLO branch detects:
pod
lesion
Severity is a derived quantity:
lesion area relative to pod area
