"""
train_currency_model.py - Transfer Learning on Real Indian Currency Dataset + Multi-Modal Data.

Trains MultiTaskMobileNetV3 on:
1. Real banknotes from Kaggle dataset (10, 20, 50, 100, 200, 500)
2. Real webcam crops of ₹50 note under incandescent lighting (augmented)
3. Synthetic banknotes across ALL denominations (10, 20, 50, 100, 200, 500, 2000)
4. Counterfeit CMYK rosette samples across all classes for counterfeit head
5. Background / Non-currency negative samples

Classes: Background (0), 10 (1), 20 (2), 50 (3), 100 (4), 200 (5), 500 (6), 2000 (7)
Dual Heads:
  1. Denomination Classification (Cross-Entropy Loss)
  2. Counterfeit Probability (Binary Cross-Entropy Loss)
"""

import os
import random
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from core.model import MultiTaskMobileNetV3, DEFAULT_DENOMINATIONS
from synthetic_data import SyntheticCurrencyGenerator


# Map Kaggle dataset YOLO class index to model denomination index
# Dataset: 0: 10, 1: 20, 2: 50, 3: 100, 4: 200, 5: 500
KAGGLE_CLASS_MAP = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500"
}

DENOM_TO_IDX = {denom: idx for idx, denom in enumerate(DEFAULT_DENOMINATIONS)}


class KaggleIndianCurrencyDataset(Dataset):
    """
    Dataset loader for real Indian currency, synthetic banknotes across all denominations,
    augmented real webcam crops, and background samples.
    """

    def __init__(
        self,
        base_dir: str = "data/kaggle_indian_currency",
        split: str = "train",
        max_samples_per_class: int = 180,
        img_size: int = 224
    ):
        self.samples: List[Tuple[np.ndarray, int, float]] = []  # (image_bgr, class_idx, is_counterfeit)
        self.img_size = img_size
        self.img_dir = os.path.join(base_dir, "images", split)
        self.lbl_dir = os.path.join(base_dir, "labels", split)

        print(f"Loading {split} dataset from {base_dir}...")
        # 1. Load real Kaggle crops for classes 10, 20, 50, 100, 200, 500
        self._load_kaggle_crops(max_samples_per_class)

        # 2. Add synthetic notes for ALL denominations (genuine and counterfeit)
        synth_per_denom = 60 if split == "train" else 15
        self._add_synthetic_across_all_denoms(count_per_denom=synth_per_denom)

        # 3. Add real webcam ₹50 augmented samples (so the model recognizes real webcam notes)
        webcam_count = 100 if split == "train" else 20
        self._add_real_webcam_50_samples(count=webcam_count)

        # 4. Add background negative samples
        bg_count = 80 if split == "train" else 20
        self._add_background_samples(count=bg_count)

        random.seed(42 if split == "train" else 99)
        random.shuffle(self.samples)
        print(f"Total {split} samples loaded: {len(self.samples)}")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5) if split == "train" else T.Lambda(lambda x: x),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10) if split == "train" else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_kaggle_crops(self, max_per_class: int):
        class_counts = {k: 0 for k in KAGGLE_CLASS_MAP.keys()}
        if not os.path.exists(self.lbl_dir):
            return

        label_files = [f for f in os.listdir(self.lbl_dir) if f.endswith(".txt")]
        for lf_name in label_files:
            txt_path = os.path.join(self.lbl_dir, lf_name)
            img_name = lf_name.replace(".txt", ".jpg")
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                img_name = lf_name.replace(".txt", ".jpeg")
                img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c_id = int(float(parts[0]))
                    if c_id not in KAGGLE_CLASS_MAP or class_counts[c_id] >= max_per_class:
                        continue

                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0, int((xc - bw / 2.0) * w))
                    y1 = max(0, int((yc - bh / 2.0) * h))
                    x2 = min(w, int((xc + bw / 2.0) * w))
                    y2 = min(h, int((yc + bh / 2.0) * h))

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                        continue

                    denom_str = KAGGLE_CLASS_MAP[c_id]
                    model_cls_idx = DENOM_TO_IDX[denom_str]

                    is_cf = False
                    if random.random() < 0.20:
                        is_cf = True
                        crop = self._inject_halftone(crop)

                    self.samples.append((crop, model_cls_idx, 1.0 if is_cf else 0.0))
                    class_counts[c_id] += 1

    def _inject_halftone(self, crop: np.ndarray) -> np.ndarray:
        """Inject realistic CMYK halftone printer rosettes into a note crop."""
        ch, cw = crop.shape[:2]
        c_copy = crop.copy()
        y, x = np.ogrid[:ch, :cw]
        th = np.radians(15.0)
        u = x * np.cos(th) + y * np.sin(th)
        v = -x * np.sin(th) + y * np.cos(th)
        dots = (np.sin(u * 0.75) * np.sin(v * 0.75)) > 0.45
        c_copy[dots] = (220, 200, 20)
        return cv2.GaussianBlur(c_copy, (3, 3), 0.5)

    def _add_synthetic_across_all_denoms(self, count_per_denom: int):
        """
        Generate synthetic banknotes across EVERY denomination (10, 20, 50, 100, 200, 500, 2000).
        This ensures the CNN learns the numerical and chromatic features of synthetic notes
        instead of associating synthetic style solely with any single denomination.
        """
        all_denoms = ["10", "20", "50", "100", "200", "500", "2000"]
        for denom in all_denoms:
            cls_idx = DENOM_TO_IDX[denom]
            for i in range(count_per_denom):
                is_cf = (i % 2 == 1)  # 50% genuine, 50% counterfeit
                note = SyntheticCurrencyGenerator.generate_banknote(
                    denomination=denom,
                    is_counterfeit=is_cf,
                    width=480,
                    height=240
                )
                self.samples.append((note, cls_idx, 1.0 if is_cf else 0.0))

    def _add_real_webcam_50_samples(self, count: int):
        """
        Augment real webcam ₹50 crops captured from the user's camera.
        Simulates warm indoor incandescent lighting, varying exposure, and angles.
        """
        cls_idx = DENOM_TO_IDX["50"]
        webcam_files = [
            "real_note_from_screenshot.jpg",
            "exact_note_crop.jpg",
            "winner_crop.jpg",
            "user_webcam_crop2.jpg",
            "real_note_crop.jpg"
        ]
        loaded_crops = []
        for wf in webcam_files:
            if os.path.exists(wf):
                im = cv2.imread(wf)
                if im is not None and im.size > 0:
                    loaded_crops.append(im)

        if not loaded_crops:
            return

        for i in range(count):
            base_crop = random.choice(loaded_crops).copy()
            # Random perspective / slight rotation
            h, w = base_crop.shape[:2]
            angle = random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, random.uniform(0.9, 1.1))
            aug_crop = cv2.warpAffine(base_crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random color temperature shift (warm indoor incandescent)
            b, g, r = cv2.split(aug_crop.astype(np.float32))
            r_scale = random.uniform(0.95, 1.25)
            g_scale = random.uniform(0.95, 1.15)
            b_scale = random.uniform(0.75, 1.05)
            aug_crop = cv2.merge([
                np.clip(b * b_scale, 0, 255).astype(np.uint8),
                np.clip(g * g_scale, 0, 255).astype(np.uint8),
                np.clip(r * r_scale, 0, 255).astype(np.uint8)
            ])

            is_cf = False
            if random.random() < 0.15:
                is_cf = True
                aug_crop = self._inject_halftone(aug_crop)

            self.samples.append((aug_crop, cls_idx, 1.0 if is_cf else 0.0))

    def _add_background_samples(self, count: int):
        """Negative background samples (desks, skin, fabric, walls, noise)."""
        cls_idx = DENOM_TO_IDX["Background"]
        for _ in range(count):
            bg_type = random.choice(["solid", "noise", "skin", "fabric"])
            if bg_type == "solid":
                bg_color = np.random.randint(30, 220, (3,))
                patch = np.full((224, 224, 3), bg_color, dtype=np.uint8)
            elif bg_type == "skin":
                # Typical human skin tones (BGR)
                patch = np.zeros((224, 224, 3), dtype=np.uint8)
                patch[:, :, 0] = np.random.randint(110, 160)  # B
                patch[:, :, 1] = np.random.randint(140, 190)  # G
                patch[:, :, 2] = np.random.randint(190, 240)  # R
            elif bg_type == "fabric":
                # Dark cushion / headboard brown
                patch = np.zeros((224, 224, 3), dtype=np.uint8)
                patch[:, :, 0] = np.random.randint(30, 70)
                patch[:, :, 1] = np.random.randint(40, 80)
                patch[:, :, 2] = np.random.randint(60, 110)
            else:
                patch = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

            noise = np.random.normal(0, 12, patch.shape).astype(np.int16)
            patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            self.samples.append((patch, cls_idx, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        crop_bgr, cls_idx, is_cf = self.samples[idx]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor, cls_idx, float(is_cf)


def train_model(epochs: int = 6, batch_size: int = 32, lr: float = 1.0e-3):
    print("=" * 65)
    print(" TRAINING MULTI-TASK MOBILENET-V3 ON REAL & SYNTHETIC DATASET")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = MultiTaskMobileNetV3(
        num_denominations=len(DEFAULT_DENOMINATIONS),
        denominations=DEFAULT_DENOMINATIONS,
        pretrained=True
    ).to(device)

    # Freeze earlier layers for CPU speed; fine-tune upper layers + dual heads
    for param in model.features[:6].parameters():
        param.requires_grad = False

    train_dataset = KaggleIndianCurrencyDataset(split="train", max_samples_per_class=180)
    val_dataset = KaggleIndianCurrencyDataset(split="val", max_samples_per_class=45)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion_denom = nn.CrossEntropyLoss()
    criterion_cf = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0
    best_weights_path = "models/currency_mobilenet_v3.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels, cf_labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            cf_labels = cf_labels.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_d = criterion_denom(outputs["denomination_logits"], labels)
            loss_c = criterion_cf(outputs["counterfeit_logits"], cf_labels)
            loss = loss_d + 0.8 * loss_c

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs["denomination_logits"], dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        train_acc = train_correct / max(1, train_total)

        # Validation pass
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs["denomination_logits"], dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / max(1, val_total)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path} ({val_acc*100:.1f}%)")

    print("\nTraining completed successfully!")
    print(f"Final Model Saved: {best_weights_path} (Best Val Accuracy: {best_val_acc*100:.1f}%)")

    # Audit predictions on real webcam crops and synthetic notes
    print("\n--- INFERENCE AUDIT ON CRITICAL SAMPLES ---")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_imgs = ["exact_note_crop.jpg", "real_note_from_screenshot.jpg", "winner_crop.jpg"]
    for img_file in test_imgs:
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform_val(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
                pred_idx = int(np.argmax(probs))
                pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
                cf_prob = float(torch.sigmoid(out["counterfeit_logits"])[0].item())
                print(f"[Real Webcam Crop] {img_file} -> Predicted: ₹{pred_denom} ({probs[pred_idx]*100:.1f}%) | Counterfeit Prob: {cf_prob:.3f}")

    # Check synthetic 500 note
    for d in ["50", "100", "500", "2000"]:
        syn_note = SyntheticCurrencyGenerator.generate_banknote(d, is_counterfeit=False, width=480, height=240)
        rgb = cv2.cvtColor(syn_note, cv2.COLOR_BGR2RGB)
        tensor = transform_val(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
            print(f"[Synthetic Note] ₹{d} -> Predicted: ₹{pred_denom} ({probs[pred_idx]*100:.1f}%)")


if __name__ == "__main__":
    train_model(epochs=6, batch_size=32, lr=1.0e-3)
