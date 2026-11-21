"""
core/model.py - Multi-Task CNN Architecture for Currency Recognition and Counterfeit Detection.

Architecture:
- Base: MobileNetV3-Small (optimized for edge/CPU inference, latency < 15ms).
- Head 1: Denomination Classification (Cross-Entropy).
- Head 2: Counterfeit Probability Estimation (Binary Cross-Entropy with Sigmoid).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    models = None
    class _DummyModule:
        pass
    nn = type("nn", (), {"Module": _DummyModule})()


DEFAULT_DENOMINATIONS: List[str] = [
    "Background",
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000"
]


class OpenCVDNNMobileNetV3:
    """
    Ultra-lightweight OpenCV DNN runtime for MultiTaskMobileNetV3 ONNX model.
    Runs on pure NumPy + OpenCV with zero PyTorch / CUDA overhead (< 65MB total bundle).
    """

    def __init__(self, onnx_path: str, denominations: Optional[List[str]] = None) -> None:
        self.denominations = denominations or DEFAULT_DENOMINATIONS
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.out_names = list(self.net.getUnconnectedOutLayersNames())

    def predict(self, bgr_img: np.ndarray) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Run inference on a single BGR image.
        Returns:
            - 'probs': dictionary mapping denomination to softmax probability
            - 'counterfeit_prob': float in [0, 1]
        """
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            resized.astype(np.float32) / 255.0,
            scalefactor=1.0,
            size=(224, 224),
            mean=(0.485, 0.456, 0.406),
            swapRB=False,
            crop=False
        )
        blob[:, 0, :, :] /= 0.229
        blob[:, 1, :, :] /= 0.224
        blob[:, 2, :, :] /= 0.225

        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        out_map = {name: val for name, val in zip(self.out_names, outs)}

        # Denomination logits
        logits = out_map["denomination_logits"][0]
        e_x = np.exp(logits - np.max(logits))
        probs_arr = e_x / max(1e-7, float(np.sum(e_x)))
        probs_dict = {
            self.denominations[i]: float(probs_arr[i])
            for i in range(min(len(self.denominations), len(probs_arr)))
        }

        # Counterfeit probability
        if "counterfeit_prob" in out_map:
            cf_prob = float(out_map["counterfeit_prob"][0][0])
        elif "counterfeit_logits" in out_map:
            logit_cf = float(out_map["counterfeit_logits"][0][0])
            cf_prob = float(1.0 / (1.0 + np.exp(-logit_cf)))
        else:
            cf_prob = 0.0

        return {
            "probs": probs_dict,
            "counterfeit_prob": cf_prob
        }


class MultiTaskMobileNetV3(nn.Module):
    """
    Multi-Task MobileNetV3 for concurrent denomination classification
    and counterfeit probability prediction.
    """

    def __init__(
        self,
        num_denominations: int = len(DEFAULT_DENOMINATIONS),
        denominations: Optional[List[str]] = None,
        pretrained: bool = False,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.denominations = denominations or DEFAULT_DENOMINATIONS
        self.num_denominations = num_denominations

        # Load MobileNetV3-Small backbone
        # We extract features and avgpool
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # In MobileNetV3-Small, feature output channels = 576
        in_features = 576

        # Shared projection layer for feature refinement
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        # Head 1: Denomination Classification (Cross-Entropy logits)
        self.head_denomination = nn.Sequential(
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, self.num_denominations)
        )

        # Head 2: Counterfeit Detection (Binary logit -> Sigmoid probability)
        # 0.0 = Genuine, 1.0 = Counterfeit
        self.head_counterfeit = nn.Sequential(
            nn.Linear(512, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self._initialize_head_weights()

    def _initialize_head_weights(self) -> None:
        """Kaiming normal initialization for task-specific projection heads."""
        for head in [self.shared_fc, self.head_denomination, self.head_counterfeit]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W) normalized to ImageNet statistics or [0, 1].
        Returns:
            Dictionary containing:
                - 'denomination_logits': Tensor of shape (B, num_denominations)
                - 'counterfeit_logits': Tensor of shape (B, 1)
                - 'counterfeit_prob': Tensor of shape (B, 1) in range [0, 1]
                - 'features': Extracted embedding tensor (B, 512)
        """
        # Backbone feature extraction
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)

        # Shared representation
        shared = self.shared_fc(flattened)

        # Task Heads
        denom_logits = self.head_denomination(shared)
        counterfeit_logit = self.head_counterfeit(shared)
        counterfeit_prob = self.sigmoid(counterfeit_logit)

        return {
            "denomination_logits": denom_logits,
            "counterfeit_logits": counterfeit_logit,
            "counterfeit_prob": counterfeit_prob,
            "features": shared
        }

    @torch.inference_mode()
    def predict_single(
        self,
        tensor_img: torch.Tensor,
        counterfeit_threshold: float = 0.50
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Inference helper for a single image tensor of shape (1, 3, H, W) or (3, H, W).
        """
        if tensor_img.dim() == 3:
            tensor_img = tensor_img.unsqueeze(0)

        out = self.forward(tensor_img)
        denom_logits = out["denomination_logits"][0]
        counterfeit_prob = out["counterfeit_prob"][0].item()

        probs = torch.softmax(denom_logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        denom_confidence = probs[pred_idx].item()
        denom_name = self.denominations[pred_idx] if pred_idx < len(self.denominations) else f"Class_{pred_idx}"

        is_counterfeit = counterfeit_prob >= counterfeit_threshold

        return {
            "denomination": denom_name,
            "denomination_idx": pred_idx,
            "confidence": denom_confidence,
            "counterfeit_prob": counterfeit_prob,
            "is_counterfeit": is_counterfeit,
            "verdict": "SUSPECT / COUNTERFEIT" if is_counterfeit else "GENUINE"
        }


class MultiTaskCurrencyLoss(nn.Module):
    """
    Joint loss function:
    L = alpha * L_denomination(CrossEntropy) + beta * L_counterfeit(BCEWithLogits)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target_denom: torch.Tensor,
        target_counterfeit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions: Output dict from MultiTaskMobileNetV3
            target_denom: LongTensor of shape (B,) with class indices
            target_counterfeit: FloatTensor of shape (B, 1) or (B,) with binary 0/1 labels
        Returns:
            Tuple of (total_loss, denom_loss, counterfeit_loss)
        """
        target_counterfeit = target_counterfeit.view(-1, 1).float()
        l_denom = self.ce_loss(predictions["denomination_logits"], target_denom)
        l_counterfeit = self.bce_loss(predictions["counterfeit_logits"], target_counterfeit)
        total_loss = self.alpha * l_denom + self.beta * l_counterfeit
        return total_loss, l_denom, l_counterfeit

