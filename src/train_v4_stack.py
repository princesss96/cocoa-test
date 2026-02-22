%%bash
cat > /content/cocoa-test/src/train_v4_stack.py <<'PY'
from __future__ import annotations

import argparse, os, json, random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_eval_tf(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def make_backbone(name: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        m.fc = torch.nn.Identity()
        return m, feat_dim
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        m.fc = torch.nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # classifier = Sequential(Dropout, Linear)
        feat_dim = m.classifier[1].in_features
        m.classifier = torch.nn.Identity()
        return m, feat_dim
    raise ValueError("backbone must be one of: resnet18, resnet50, efficientnet_b0")

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model(x)
        f = f.detach().float().cpu().numpy()
        feats.append(f)
        ys.append(y.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

def metrics_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str, help="root with train/val/test")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18","resnet50","efficientnet_b0"])
    ap.add_argument("--pca_dim", type=int, default=256, help="PCA components (fit on train only)")
    ap.add_argument("--out_dir", type=str, default="runs_cls_v4_stack")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_eval_tf(args.img_size)

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=tf)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    backbone, feat_dim = make_backbone(args.backbone)
    backbone = backbone.to(device)

    print(f"Backbone={args.backbone} feat_dim={feat_dim}")
    print("Extracting features...")
    Xtr, ytr = extract_features(backbone, train_loader, device)
    Xva, yva = extract_features(backbone, val_loader, device)
    Xte, yte = extract_features(backbone, test_loader, device)

    # PCA fit on TRAIN only (avoid leakage)
    pca = PCA(n_components=min(args.pca_dim, Xtr.shape[1]), random_state=args.seed)
    Xtr_p = pca.fit_transform(Xtr)
    Xva_p = pca.transform(Xva)
    Xte_p = pca.transform(Xte)
    print("PCA explained variance ratio sum:", float(np.sum(pca.explained_variance_ratio_)))

    # Stacking (balanced)
    base = [
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", C=3.0, gamma="scale")),
        ("rf",  RandomForestClassifier(n_estimators=400, random_state=args.seed, class_weight="balanced_subsample")),
        ("lr",  LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)),
    ]
    meta = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)

    clf = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    print("Training stacking classifier...")
    clf.fit(Xtr_p, ytr)

    # Optional: report VAL
    yva_pred = clf.predict(Xva_p)
    val_m = metrics_dict(yva, yva_pred)
    print("\nVal metrics:", {k:v for k,v in val_m.items() if k!="confusion_matrix"})
    print("Val confusion matrix:\n", np.array(val_m["confusion_matrix"]))

    # TEST
    yte_pred = clf.predict(Xte_p)
    test_m = metrics_dict(yte, yte_pred)
    print("\nTest metrics:", {k:v for k,v in test_m.items() if k!="confusion_matrix"})
    print("Confusion matrix:\n", np.array(test_m["confusion_matrix"]))

    # Save artifacts
    np.save(os.path.join(args.out_dir, "pca_components.npy"), pca.components_)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({
            "seed": args.seed,
            "backbone": args.backbone,
            "img_size": args.img_size,
            "pca_dim": args.pca_dim,
            "class_to_idx": train_ds.class_to_idx,
            "val_metrics": val_m,
            "test_metrics": test_m,
        }, f, indent=2)

if __name__ == "__main__":
    main()
PY