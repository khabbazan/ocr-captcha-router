import json
import re
import numpy as np
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import (
    AutoModelForImageClassification,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


class CaptchaResolver:
    # Route a captcha to digit/math TrOCR experts with a safe fallback.

    def __init__(
        self, repo_dir, token=None, device=None,
        math_confidence_threshold=None,
    ):
        self.repo_dir = Path(repo_dir)
        self.token = token
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = json.loads(
            (self.repo_dir / "pipeline_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.threshold = float(
            self.config.get("router_confidence_threshold", 0.95)
        )
        configured_math_threshold = float(
            self.config.get(
                "math_confidence_threshold",
                self.config.get(
                    "math_conflict_probability_threshold", 0.85
                ),
            )
        )
        self.math_confidence_threshold = float(
            configured_math_threshold
            if math_confidence_threshold is None
            else math_confidence_threshold
        )
        if not 0.0 <= self.math_confidence_threshold <= 1.0:
            raise ValueError(
                "math_confidence_threshold must be between 0.0 and 1.0"
            )

        router_dir = self.repo_dir / self.config["router_path"]
        preprocess = json.loads(
            (router_dir / "router_preprocess.json").read_text(encoding="utf-8")
        )
        self.router_height = int(preprocess["input_height"])
        self.router_width = int(preprocess["input_width"])
        self.router_scale = float(preprocess["eval_scale"])
        self.router_mean = np.array(preprocess["mean"], dtype=np.float32)
        self.router_std = np.array(preprocess["std"], dtype=np.float32)
        self.router_model = (
            AutoModelForImageClassification.from_pretrained(router_dir)
            .to(self.device)
            .eval()
        )

        self._experts = {}
        self.full_math = re.compile(self.config["full_math_pattern"])
        self.partial_math = re.compile(
            self.config["partial_math_pattern"]
        )
        self.digits_only = re.compile(self.config["digit_pattern"])

    @classmethod
    def from_pretrained(
        cls, repo_id, token=None, device=None,
        math_confidence_threshold=None,
    ):
        repo_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=[
                "config.json",
                "router_preprocess.json",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model-*.bin",
                "pytorch_model.bin.index.json",
                "pipeline_config.json",
                "resolver.py",
                "README.md",
            ],
        )
        return cls(
            repo_dir=repo_dir,
            token=token,
            device=device,
            math_confidence_threshold=math_confidence_threshold,
        )

    @staticmethod
    def normalize(text):
        return (
            str(text)
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(" ", "")
            .strip()
        )

    def router_tensor(self, image):
        # Aspect-preserving letterbox into (router_height x router_width),
        # padded with the median corner colour. Must match router_preprocess.json.
        image = image.convert("RGB")
        width, height = image.size
        corners = np.array(
            [
                image.getpixel((0, 0)),
                image.getpixel((width - 1, 0)),
                image.getpixel((0, height - 1)),
                image.getpixel((width - 1, height - 1)),
            ],
            dtype=np.uint8,
        )
        background = tuple(
            np.median(corners, axis=0).astype(np.uint8).tolist()
        )
        scale = min(
            self.router_width / width, self.router_height / height
        ) * self.router_scale
        new_width = max(8, round(width * scale))
        new_height = max(8, round(height * scale))
        canvas = Image.new(
            "RGB", (self.router_width, self.router_height), background
        )
        canvas.paste(
            image.resize((new_width, new_height), Image.Resampling.LANCZOS),
            (
                (self.router_width - new_width) // 2,
                (self.router_height - new_height) // 2,
            ),
        )
        array = (
            np.asarray(canvas, dtype=np.float32) / 255.0 - self.router_mean
        ) / self.router_std
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def repair_math(self, text):
        text = self.normalize(text)
        if re.fullmatch(r"\d+[+-]\d+=", text):
            return text + "?"
        return text

    def _load_expert(self, route):
        if route not in self._experts:
            key = "math_model_id" if route == "math" else "digit_model_id"
            model_id = self.config[key]
            processor = TrOCRProcessor.from_pretrained(
                model_id, token=self.token
            )
            model = (
                VisionEncoderDecoderModel.from_pretrained(
                    model_id, token=self.token
                )
                .to(self.device)
                .eval()
            )
            self._experts[route] = (processor, model)
        return self._experts[route]

    def route(self, image):
        values = self.router_tensor(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(
                self.router_model(pixel_values=values).logits, dim=-1
            )[0]
        route_id = int(probabilities.argmax().item())
        id2label = self.router_model.config.id2label
        label = id2label.get(route_id, id2label.get(str(route_id)))
        scores = {
            id2label.get(i, id2label.get(str(i))): float(value.item())
            for i, value in enumerate(probabilities)
        }
        return label, float(probabilities[route_id].item()), scores

    def _digit_views(self, image):
        original = image.convert("RGB")
        gray = ImageOps.autocontrast(original.convert("L"))
        views = [original, gray.convert("RGB")]
        if float(np.asarray(gray, dtype=np.float32).mean()) < 128.0:
            views.append(ImageOps.invert(gray).convert("RGB"))
        return views

    def run_expert(self, image, route):
        processor, model = self._load_expert(route)
        if route == "math":
            values = processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(self.device)
            with torch.inference_mode():
                generated = model.generate(
                    values, num_beams=4, max_length=32
                )
            decoded = processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            return self.normalize(decoded)

        # No token whitelist: constraining the decoder to digit tokens made it
        # append spurious digits (e.g. 9404 -> 9404404). Validate with the
        # digit_pattern regex instead and pick the best valid view.
        candidates = []
        for view in self._digit_views(image):
            values = processor(
                images=view, return_tensors="pt"
            ).pixel_values.to(self.device)
            generation_kwargs = {
                "num_beams": 4,
                "max_length": 16,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            with torch.inference_mode():
                output = model.generate(values, **generation_kwargs)
            decoded = self.normalize(
                processor.batch_decode(
                    output.sequences, skip_special_tokens=True
                )[0]
            )
            score = float(output.sequences_scores[0].item())
            candidates.append((decoded, score))

        valid = [
            candidate for candidate in candidates
            if self.digits_only.fullmatch(candidate[0])
        ]
        return max(valid or candidates, key=lambda item: item[1])[0]

    def choose_candidates(
        self, digits_text, math_text, raw_route, probabilities
    ):
        digits_text = self.normalize(digits_text)
        math_text = self.repair_math(math_text)

        digits_valid = bool(self.digits_only.fullmatch(digits_text))
        math_valid = bool(self.full_math.fullmatch(math_text))
        math_partial = bool(self.partial_math.fullmatch(math_text))
        math_probability = float(probabilities.get("math", 0.0))

        if digits_valid and math_valid:
            if (
                raw_route == "math"
                and math_probability >= self.math_confidence_threshold
            ):
                return (
                    "math", math_text, True,
                    "both valid; calibrated router strongly supports math",
                )
            return (
                "digits", digits_text, True,
                "both valid; preserve constrained digit candidate",
            )

        if digits_valid:
            return "digits", digits_text, True, "digit candidate is valid"
        if math_valid:
            return "math", math_text, True, "math candidate is valid"

        # A math expert can hallucinate '-' or '=' from grid lines. A partial
        # equation must never turn a digit-routed image into a false math result.
        reason = (
            "partial math candidate rejected"
            if math_partial
            else "neither candidate matched its grammar"
        )
        return "unknown", digits_text or math_text, False, reason

    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        raw_route, confidence, probabilities = self.route(image)
        primary_text = self.run_expert(image, raw_route)
        if raw_route == "math":
            primary_text = self.repair_math(primary_text)
            primary_valid = bool(self.full_math.fullmatch(primary_text))
        else:
            primary_valid = bool(self.digits_only.fullmatch(primary_text))

        use_fallback = confidence < self.threshold or not primary_valid
        if not use_fallback:
            return {
                "text": primary_text,
                "route": raw_route,
                "raw_router_route": raw_route,
                "router_confidence": confidence,
                "router_probabilities": probabilities,
                "math_confidence_threshold": self.math_confidence_threshold,
                "used_dual_model_fallback": False,
                "selection_reason": "high-confidence router and valid primary output",
                "valid_format": True,
            }

        if raw_route == "math":
            math_text = primary_text
            digits_text = self.run_expert(image, "digits")
        else:
            digits_text = primary_text
            math_text = self.run_expert(image, "math")

        route, text, valid, selection_reason = self.choose_candidates(
            digits_text,
            math_text,
            raw_route,
            probabilities,
        )
        return {
            "text": text,
            "route": route,
            "raw_router_route": raw_route,
            "router_confidence": confidence,
            "router_probabilities": probabilities,
            "math_confidence_threshold": self.math_confidence_threshold,
            "used_dual_model_fallback": True,
            "digits_candidate": digits_text,
            "math_candidate": math_text,
            "selection_reason": selection_reason,
            "valid_format": valid,
        }

    __call__ = predict
