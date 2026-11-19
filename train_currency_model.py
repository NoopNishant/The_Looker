"""
train_currency_model.py - Rigorous Multi-Task Transfer Learning on Indian Currency + Hard Negatives.

Trains MultiTaskMobileNetV3 for ~10 minutes covering every real-world scenario:
1. Real banknotes from Kaggle dataset across all classes: 10, 20, 50, 100, 200, 500
2. Real webcam crops of ₹50 note under incandescent lighting with finger occlusions
3. Synthetic banknotes across ALL denominations (10, 20, 50, 100, 200, 500, 2000)
4. Counterfeit CMYK rosette samples across all classes for counterfeit head
5. RIGOROUS BACKGROUND / HARD NEGATIVES (Human faces, skin, hands, clothes, rooms, Kaggle background crops, printed text)
6. Real-world physical degradations: perspective tilt, folds, crinkling, sensor noise, motion blur, color temperature shifts.

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
    augmented real webcam crops, and extensive hard negative background/face samples.
    """

    def __init__(
        self,
        base_dir: str = "data/kaggle_indian_currency",
        split: str = "train",
        max_samples_per_class: int = 220,
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
        synth_per_denom = 80 if split == "train" else 20
        self._add_synthetic_across_all_denoms(count_per_denom=synth_per_denom)

        # 3. Add real webcam ₹50 augmented samples (with hand occlusions and warm lighting)
        webcam_count = 160 if split == "train" else 35
        self._add_real_webcam_50_samples(count=webcam_count)

        # 3b. Add real & augmented toy/counterfeit note samples ('Children Bank of India', 'Full of Fun', 'Churan Label')
        fake_toy_count = 160 if split == "train" else 35
        self._add_fake_toy_samples(count=fake_toy_count)

        # 4. Add RIGOROUS background negative samples (faces, skin, hands, clothes, Kaggle backgrounds)
        bg_count = 600 if split == "train" else 120
        self._add_background_samples(count=bg_count)

        random.seed(42 if split == "train" else 99)
        random.shuffle(self.samples)
        print(f"Total {split} samples loaded: {len(self.samples)}")

        # Multi-scenario physical camera augmentations
        if split == "train":
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.92, 1.08)),
                T.RandomPerspective(distortion_scale=0.15, p=0.30),
                T.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.28, hue=0.08),
                T.ToTensor(),
                T.RandomErasing(p=0.20, scale=(0.02, 0.12), value="random"),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_kaggle_crops(self, max_per_class: int):
        class_counts = {k: 0 for k in KAGGLE_CLASS_MAP.keys()}
        if not os.path.exists(self.lbl_dir):
            return

        label_files = [f for f in os.listdir(self.lbl_dir) if f.endswith(".txt")]
        random.seed(77)
        random.shuffle(label_files)

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
                    if random.random() < 0.25:
                        is_cf = True
                        crop = self._inject_halftone(crop)

                    self.samples.append((crop, model_cls_idx, 1.0 if is_cf else 0.0))
                    class_counts[c_id] += 1

    def _inject_halftone(self, crop: np.ndarray) -> np.ndarray:
        """Inject realistic CMYK halftone printer rosettes into a note crop."""
        ch, cw = crop.shape[:2]
        c_copy = crop.copy()
        y, x = np.ogrid[:ch, :cw]
        th = np.radians(random.choice([15.0, 45.0, 75.0]))
        u = x * np.cos(th) + y * np.sin(th)
        v = -x * np.sin(th) + y * np.cos(th)
        dots = (np.sin(u * 0.75) * np.sin(v * 0.75)) > 0.42
        c_copy[dots] = (random.randint(190, 230), random.randint(180, 210), random.randint(15, 35))
        return cv2.GaussianBlur(c_copy, (3, 3), 0.5)

    def _add_synthetic_across_all_denoms(self, count_per_denom: int):
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
                # Apply random wear and crinkle
                if random.random() < 0.30:
                    note = self._apply_physical_wear(note)

                self.samples.append((note, cls_idx, 1.0 if is_cf else 0.0))

    def _apply_physical_wear(self, note: np.ndarray) -> np.ndarray:
        """Simulate physical creases, minor folds, and lighting falloff on a note."""
        nh, nw = note.shape[:2]
        out = note.copy()
        # Random fold line
        fy = random.randint(int(nh * 0.2), int(nh * 0.8))
        cv2.line(out, (0, fy), (nw, fy), (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80)), 1)
        # Lighting gradient falloff across width
        grad = np.linspace(random.uniform(0.85, 1.0), random.uniform(0.95, 1.15), nw)
        out = np.clip(out.astype(np.float32) * grad[None, :, None], 0, 255).astype(np.uint8)
        return out

    def _add_real_webcam_50_samples(self, count: int):
        cls_idx = DENOM_TO_IDX["50"]
        webcam_files = [
            "data/real_crops_50/real_note_50.jpg",
            "data/real_crops_50/real_note_50_held.jpg",
            "data/real_crops_50/real_note_50_shot2.jpg",
            "clean_note.jpg"
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
            h, w = base_crop.shape[:2]
            angle = random.uniform(-14, 14)
            scale = random.uniform(0.88, 1.12)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            aug_crop = cv2.warpAffine(base_crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Webcam incandescent warmth shift
            b, g, r = cv2.split(aug_crop.astype(np.float32))
            r_scale = random.uniform(0.95, 1.25)
            g_scale = random.uniform(0.95, 1.15)
            b_scale = random.uniform(0.75, 1.05)
            aug_crop = cv2.merge([
                np.clip(b * b_scale, 0, 255).astype(np.uint8),
                np.clip(g * g_scale, 0, 255).astype(np.uint8),
                np.clip(r * r_scale, 0, 255).astype(np.uint8)
            ])

            # Partial finger occlusion simulation
            if random.random() < 0.25:
                # Add finger-like skin ellipse on border
                fw = random.randint(15, 35)
                fh = random.randint(25, 55)
                fx = random.choice([random.randint(5, 40), random.randint(w - 45, w - 10)])
                fy = random.randint(10, h - 20)
                cv2.ellipse(aug_crop, (fx, fy), (fw, fh), random.randint(-20, 20), 0, 360, (random.randint(100, 140), random.randint(120, 160), random.randint(160, 210)), -1)

            is_cf = False
            if random.random() < 0.18:
                is_cf = True
                aug_crop = self._inject_halftone(aug_crop)

            self.samples.append((aug_crop, cls_idx, 1.0 if is_cf else 0.0))

    def _add_fake_toy_samples(self, count: int):
        """
        Add real and augmented toy/counterfeit note samples ('Children Bank of India', 'Full of Fun', 'Churan Label').
        These notes have Mahatma Gandhi printed on them but feature spoofed bank headers and imitation security threads.
        Must be rigorously flagged by the counterfeit classification head (is_cf = 1.0).
        """
        cls_idx = DENOM_TO_IDX["50"]
        fake_files = [
            "uploaded_fake_note.jpg"
        ]
        loaded_fakes = []
        for ff in fake_files:
            if os.path.exists(ff):
                im = cv2.imread(ff)
                if im is not None and im.size > 0:
                    loaded_fakes.append(im)

        if not loaded_fakes:
            return

        for i in range(count):
            base = random.choice(loaded_fakes).copy()
            h, w = base.shape[:2]
            angle = random.uniform(-14, 14)
            scale = random.uniform(0.88, 1.12)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            aug = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            b, g, r = cv2.split(aug.astype(np.float32))
            r_scale = random.uniform(0.90, 1.20)
            g_scale = random.uniform(0.90, 1.15)
            b_scale = random.uniform(0.85, 1.15)
            aug = cv2.merge([
                np.clip(b * b_scale, 0, 255).astype(np.uint8),
                np.clip(g * g_scale, 0, 255).astype(np.uint8),
                np.clip(r * r_scale, 0, 255).astype(np.uint8)
            ])

            # Injected with 100% Counterfeit ground truth (is_cf = 1.0)
            self.samples.append((aug, cls_idx, 1.0))

    def _add_background_samples(self, count: int):
        """
        Extensive hard negative background samples:
        1. Real face, hair, and head crops (from webcam screenshots)
        2. Real hand, skin, clothing, and room environment crops
        3. Real Kaggle non-banknote background regions (tables, floors, beds, walls, clutter)
        4. Synthetic human facial portraits (eyes, nose, mouth, hair, glasses)
        5. Synthetic documents, plain printed text, invoices, and household textures
        """
        cls_idx = DENOM_TO_IDX["Background"]

        # 1. Load Real Webcam Negative Samples (User's face, shirts, hands, headboard)
        real_neg_dir = "data/negative_samples"
        loaded_negs = []
        if os.path.exists(real_neg_dir):
            for fname in os.listdir(real_neg_dir):
                fpath = os.path.join(real_neg_dir, fname)
                im = cv2.imread(fpath)
                if im is not None and im.size > 0:
                    loaded_negs.append(im)

        # 2. Extract Real Kaggle Non-Banknote Background Crops
        kaggle_bgs = []
        if os.path.exists(self.lbl_dir):
            all_txts = [f for f in os.listdir(self.lbl_dir) if f.endswith(".txt")]
            random.seed(101)
            random.shuffle(all_txts)
            for lf in all_txts[:100]:
                im_path = os.path.join(self.img_dir, lf.replace(".txt", ".jpg"))
                if not os.path.exists(im_path):
                    im_path = os.path.join(self.img_dir, lf.replace(".txt", ".jpeg"))
                if not os.path.exists(im_path):
                    continue
                full_im = cv2.imread(im_path)
                if full_im is None:
                    continue
                fh, fw = full_im.shape[:2]
                if fh > 80 and fw > 80:
                    c1 = full_im[0:int(fh * 0.32), :]
                    c2 = full_im[int(fh * 0.68):fh, :]
                    if c1.shape[0] > 30 and c1.shape[1] > 30:
                        kaggle_bgs.append(c1)
                    if c2.shape[0] > 30 and c2.shape[1] > 30:
                        kaggle_bgs.append(c2)

        for i in range(count):
            sub_type = i % 5

            if sub_type == 0 and loaded_negs:
                # Augmented real face/head/shirt/hand crop
                base = random.choice(loaded_negs).copy()
                h, w = base.shape[:2]
                angle = random.uniform(-25, 25)
                scale = random.uniform(0.85, 1.25)
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
                aug = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                aug = cv2.resize(aug, (224, 224))
                self.samples.append((aug, cls_idx, 0.0))

            elif sub_type == 1 and kaggle_bgs:
                # Real Kaggle scene background (tables, floors, beds, outdoor)
                base = random.choice(kaggle_bgs)
                h, w = base.shape[:2]
                ch = min(h, random.randint(30, h))
                cw = min(w, random.randint(30, w))
                y1 = random.randint(0, h - ch)
                x1 = random.randint(0, w - cw)
                patch = cv2.resize(base[y1:y1+ch, x1:x1+cw], (224, 224))
                self.samples.append((patch, cls_idx, 0.0))

            elif sub_type == 2:
                # Synthetic Human Face / Portrait Pattern
                # Explicitly draws eyes, nose, mouth, hair, and glasses on a skin canvas
                patch = np.zeros((224, 224, 3), dtype=np.uint8)
                skin_b = random.randint(95, 165)
                skin_g = random.randint(130, 195)
                skin_r = random.randint(180, 245)
                patch[:, :] = (skin_b, skin_g, skin_r)

                # Head oval
                cv2.ellipse(patch, (112, 120), (75, 95), 0, 0, 360, (skin_b - 15, skin_g - 15, skin_r - 15), -1)
                # Hair (dark brown / black)
                cv2.ellipse(patch, (112, 50), (80, 45), 0, 0, 360, (random.randint(15, 45), random.randint(15, 40), random.randint(20, 50)), -1)
                # Eyes (left and right)
                cv2.circle(patch, (82, 105), 10, (230, 230, 235), -1)
                cv2.circle(patch, (142, 105), 10, (230, 230, 235), -1)
                cv2.circle(patch, (82, 105), 5, (20, 20, 20), -1)
                cv2.circle(patch, (142, 105), 5, (20, 20, 20), -1)
                # Eyebrows
                cv2.line(patch, (68, 92), (98, 92), (30, 25, 25), 3)
                cv2.line(patch, (126, 92), (156, 92), (30, 25, 25), 3)
                # Nose
                cv2.line(patch, (112, 110), (108, 138), (skin_b - 35, skin_g - 35, skin_r - 35), 2)
                cv2.line(patch, (108, 138), (118, 138), (skin_b - 35, skin_g - 35, skin_r - 35), 2)
                # Mouth / lips
                cv2.line(patch, (92, 165), (132, 165), (max(0, skin_b - 30), max(0, skin_g - 50), max(0, skin_r + 10)), 3)
                # Glasses frame
                if random.random() < 0.65:
                    cv2.rectangle(patch, (68, 93), (98, 117), (20, 20, 20), 2)
                    cv2.rectangle(patch, (126, 93), (156, 117), (20, 20, 20), 2)
                    cv2.line(patch, (98, 105), (126, 105), (20, 20, 20), 2)

                noise = np.random.normal(0, 10, patch.shape).astype(np.int16)
                patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                self.samples.append((patch, cls_idx, 0.0))

            elif sub_type == 3:
                # Synthetic Printed Document / Book Text (Black text on paper, NOT currency)
                patch = np.full((224, 224, 3), random.randint(220, 250), dtype=np.uint8)
                for line_y in range(25, 210, 18):
                    txt_len = random.randint(15, 35)
                    txt = "lorem ipsum dolor sit amet "[:txt_len] * 2
                    cv2.putText(patch, txt, (15, line_y), cv2.FONT_HERSHEY_PLAIN, 0.9, (35, 35, 35), 1)
                self.samples.append((patch, cls_idx, 0.0))

            else:
                # Clothing / Fabric / Room clutter / Keyboard
                fab_color = np.random.randint(40, 210, (3,))
                patch = np.full((224, 224, 3), fab_color, dtype=np.uint8)
                if random.random() < 0.5:
                    for sy in range(0, 224, 20):
                        patch[sy:sy+8, :] = np.clip(fab_color - 35, 0, 255)
                noise = np.random.normal(0, 15, patch.shape).astype(np.int16)
                patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                self.samples.append((patch, cls_idx, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        crop_bgr, cls_idx, is_cf = self.samples[idx]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor, cls_idx, float(is_cf)


def train_model(
    target_duration_minutes: float = 10.0,
    min_epochs: int = 18,
    max_epochs: int = 26,
    batch_size: int = 32,
    lr: float = 7.5e-4
):
    print("=" * 72)
    print(f" RIGOROUS MULTI-TASK TRAINING FOR EVERY SCENARIO (~{target_duration_minutes:.0f} MINUTES)")
    print("=" * 72)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    print(f"Device: {device} | PyTorch Threads: {torch.get_num_threads()}")

    target_duration_sec = target_duration_minutes * 60.0

    model = MultiTaskMobileNetV3(
        num_denominations=len(DEFAULT_DENOMINATIONS),
        denominations=DEFAULT_DENOMINATIONS,
        pretrained=True
    ).to(device)

    # Fine-tune upper feature blocks and dual classification/counterfeit heads
    for param in model.features[:5].parameters():
        param.requires_grad = False

    train_dataset = KaggleIndianCurrencyDataset(split="train", max_samples_per_class=220)
    val_dataset = KaggleIndianCurrencyDataset(split="val", max_samples_per_class=45)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion_denom = nn.CrossEntropyLoss(label_smoothing=0.04)
    criterion_cf = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1.5e-5)

    os.makedirs("models", exist_ok=True)
    best_val_score = 0.0
    best_val_acc = 0.0
    best_val_cf_acc = 0.0
    best_weights_path = "models/currency_mobilenet_v3.pth"

    start_time = time.time()
    completed_epochs = 0

    print(f"\nStarting {target_duration_minutes:.0f}-minute scheduled training curriculum...")
    print("-" * 72)

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.time()
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
            loss = loss_d + 0.85 * loss_c

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
        val_cf_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, cf_labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                cf_target = cf_labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs["denomination_logits"], dim=1)
                val_correct += (preds == labels).sum().item()

                cf_pred_bin = (outputs["counterfeit_logits"].squeeze() > 0.0).float()
                val_cf_correct += (cf_pred_bin == cf_target).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / max(1, val_total)
        val_cf_acc = val_cf_correct / max(1, val_total)
        val_score = 0.60 * val_acc + 0.40 * val_cf_acc

        t_epoch_end = time.time()
        epoch_sec = t_epoch_end - t_epoch_start
        total_elapsed_sec = t_epoch_end - start_time
        completed_epochs = epoch

        elapsed_min = int(total_elapsed_sec // 60)
        elapsed_rem_sec = int(total_elapsed_sec % 60)

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"(Epoch: {epoch_sec:.1f}s | Elapsed: {elapsed_min:02d}m {elapsed_rem_sec:02d}s) - "
            f"Loss: {train_loss/train_total:.4f} | "
            f"Train Acc: {train_acc*100:.1f}% | "
            f"Val Denom Acc: {val_acc*100:.1f}% | "
            f"Val CF Acc: {val_cf_acc*100:.1f}%"
        )

        if val_score >= best_val_score:
            best_val_score = val_score
            best_val_acc = val_acc
            best_val_cf_acc = val_cf_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  [Checkpoint] Saved best weights (Denom: {val_acc*100:.1f}%, CF: {val_cf_acc*100:.1f}%) to {best_weights_path}")

        # Stop condition: completed at least min_epochs and reached target duration
        if total_elapsed_sec >= target_duration_sec and epoch >= min_epochs:
            print(f"\n[Timer] Target training duration reached ({elapsed_min}m {elapsed_rem_sec}s). Concluding training.")
            break

    total_training_time = time.time() - start_time
    total_min = int(total_training_time // 60)
    total_sec = int(total_training_time % 60)

    print("\n" + "=" * 72)
    print(f" TRAINING COMPLETED: {completed_epochs} Epochs in {total_min}m {total_sec}s")
    print(f" Best Checkpoint Saved: {best_weights_path} (Validation Accuracy: {best_val_acc*100:.1f}%)")
    print("=" * 72)

    # Rigorous Inference Audit on All Scenarios
    print("\n" + "-" * 72)
    print(" RIGOROUS INFERENCE AUDIT ON ALL SCENARIOS")
    print("-" * 72)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. SCENARIO: Human Face Hard Negative Rejection
    print("\n[Scenario 1: Human Face Rejection]")
    for face_file in ["face_test.jpg", "data/negative_samples/user_face_0.jpg", "data/negative_samples/crop_face_0.jpg"]:
        if os.path.exists(face_file):
            f_img = cv2.imread(face_file)
            rgb_f = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
            tensor_f = transform_val(rgb_f).unsqueeze(0).to(device)
            with torch.no_grad():
                out_f = model(tensor_f)
                probs_f = torch.softmax(out_f["denomination_logits"], dim=-1)[0].numpy()
                pred_idx = int(np.argmax(probs_f))
                pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
                bg_prob = probs_f[0] * 100.0
                status = "PASS" if pred_denom == "Background" else "FAIL"
                print(f"  [{status}] {face_file} -> Predicted: {pred_denom} (Background prob: {bg_prob:.1f}%)")

    # 2. SCENARIO: Real Webcam Banknote Crops
    print("\n[Scenario 2: Real Webcam ₹50 Note Crops]")
    test_crops = ["data/real_crops_50/real_note_50.jpg", "data/real_crops_50/real_note_50_held.jpg", "clean_note.jpg"]
    for img_file in test_crops:
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
                status = "PASS" if pred_denom == "50" else "FAIL"
                print(f"  [{status}] {img_file} -> Predicted: ₹{pred_denom} ({probs[pred_idx]*100:.1f}%) | CF Prob: {cf_prob:.3f}")

    # 2b. SCENARIO: Fake Toy Banknote Forensics (Children Bank of India / Churan Label)
    print("\n[Scenario 2b: Fake Toy Banknote Forensics (Children Bank of India / Churan Label)]")
    fake_notes = ["uploaded_fake_note.jpg"]
    for fn in fake_notes:
        if os.path.exists(fn):
            img = cv2.imread(fn)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform_val(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
                pred_idx = int(np.argmax(probs))
                pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
                cf_prob = float(torch.sigmoid(out["counterfeit_logits"])[0].item())
                status = "PASS" if cf_prob >= 0.60 else "FAIL"
                print(f"  [{status}] {fn} -> Denom: ₹{pred_denom} ({probs[pred_idx]*100:.1f}%) | Counterfeit Prob: {cf_prob:.3f}")

    # 3. SCENARIO: All 7 Indian Rupee Denominations (Genuine Notes)
    print("\n[Scenario 3: Genuine Indian Currency Notes (All Denominations)]")
    for d in ["10", "20", "50", "100", "200", "500", "2000"]:
        syn_note = SyntheticCurrencyGenerator.generate_banknote(denomination=d, is_counterfeit=False, width=480, height=240)
        rgb = cv2.cvtColor(syn_note, cv2.COLOR_BGR2RGB)
        tensor = transform_val(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
            cf_prob = float(torch.sigmoid(out["counterfeit_logits"])[0].item())
            status = "PASS" if pred_denom == d and cf_prob < 0.50 else "WARN"
            print(f"  [{status}] ₹{d} -> Predicted: ₹{pred_denom} ({probs[pred_idx]*100:.1f}%) | CF Prob: {cf_prob:.3f}")

    # 4. SCENARIO: All 7 Indian Rupee Denominations (Counterfeit CMYK Rosettes)
    print("\n[Scenario 4: Counterfeit CMYK Rosette Notes (All Denominations)]")
    for d in ["10", "20", "50", "100", "200", "500", "2000"]:
        syn_note = SyntheticCurrencyGenerator.generate_banknote(denomination=d, is_counterfeit=True, width=480, height=240)
        rgb = cv2.cvtColor(syn_note, cv2.COLOR_BGR2RGB)
        tensor = transform_val(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
            cf_prob = float(torch.sigmoid(out["counterfeit_logits"])[0].item())
            status = "PASS" if cf_prob >= 0.50 else "WARN"
            print(f"  [{status}] ₹{d} Counterfeit -> Predicted: ₹{pred_denom} | CF Prob: {cf_prob:.3f}")

    # 5. SCENARIO: Hard Negative Room Clutter & Fabrics
    print("\n[Scenario 5: Room Clutter, Fabrics, & Hands]")
    for neg_f in ["data/negative_samples/user_shirt_0.jpg", "data/negative_samples/user_hand_0.jpg", "data/negative_samples/user_cushion_0.jpg"]:
        if os.path.exists(neg_f):
            im = cv2.imread(neg_f)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tensor = transform_val(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out["denomination_logits"], dim=-1)[0].numpy()
                pred_idx = int(np.argmax(probs))
                pred_denom = DEFAULT_DENOMINATIONS[pred_idx]
                status = "PASS" if pred_denom == "Background" else "WARN"
                print(f"  [{status}] {neg_f} -> Predicted: {pred_denom} ({probs[0]*100:.1f}% Background)")

    print("\n" + "=" * 72)
    print(" ALL SCENARIO AUDIT COMPLETE!")
    print("=" * 72)


if __name__ == "__main__":
    train_model(target_duration_minutes=3.0, min_epochs=8, max_epochs=10, batch_size=32, lr=7.5e-4)
