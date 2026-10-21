
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification


class CaptchaRouter:
    """P(math) router. Usage: CaptchaRouter('path/to/captcha_router_v3').route(PIL.Image)."""

    def __init__(self, directory, device=None):
        directory = Path(directory)
        cfg = json.loads((directory / "router_preprocess.json").read_text())
        self.h, self.w, self.scale = cfg["input_height"], cfg["input_width"], cfg["eval_scale"]
        self.mean = np.array(cfg["mean"], dtype=np.float32)
        self.std = np.array(cfg["std"], dtype=np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(directory).to(self.device).eval()

    def _prepare(self, image):
        image = image.convert("RGB")
        width, height = image.size
        corners = np.array([image.getpixel(p) for p in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))])
        background = tuple(np.median(corners, axis=0).astype(np.uint8).tolist())
        scale = min(self.w / width, self.h / height) * self.scale
        new_w, new_h = max(8, round(width * scale)), max(8, round(height * scale))
        canvas = Image.new("RGB", (self.w, self.h), background)
        canvas.paste(image.resize((new_w, new_h), Image.Resampling.LANCZOS), ((self.w - new_w) // 2, (self.h - new_h) // 2))
        array = (np.asarray(canvas, dtype=np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    @torch.no_grad()
    def route(self, image):
        batch = self._prepare(image).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(pixel_values=batch).logits.float(), dim=-1)[0].cpu().numpy()
        label = "math" if probabilities[1] >= 0.5 else "digits"
        return label, float(probabilities.max()), {"digits": float(probabilities[0]), "math": float(probabilities[1])}
