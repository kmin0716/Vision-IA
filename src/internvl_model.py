# Module pour charger, configurer et utiliser le modèle InternVL3.5 pour l’analyse d’images et de texte.

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ⚠️ Force le tokenizer "lent" (évite le bug Qwen2TokenizerFast)
os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-1B"

class InternVLAnalyzer:
    """
    Loader + analyse InternVL3.5-1B.
    Inclut un patch anti-bug si jamais un tokenizer Qwen2 'fast' se glisse quand même.
    """
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model.to("cpu")

        # 🔧 Garde-fou: si un Qwen2 "fast" apparait sans attributs image, on les ajoute.
        tok = getattr(self.processor, "tokenizer", None)
        try:
            name_or_path = str(getattr(tok, "name_or_path", "")).lower()
        except Exception:
            name_or_path = ""
        if tok is not None and ("qwen" in name_or_path or "qwen2" in name_or_path):
            if not hasattr(tok, "start_image_token"):
                setattr(tok, "start_image_token", "<image>")
            if not hasattr(tok, "end_image_token"):
                setattr(tok, "end_image_token", "</image>")

    def analyze_image(self, image_path: Path, question: str, max_new_tokens: int = 256) -> str:
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": image},          # passe l’objet PIL
                {"type": "text", "text": question}
            ]}
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    def analyze_batch(self, image_paths: Iterable[Path], question: str) -> List[Dict[str, Any]]:
        results = []
        for idx, p in enumerate(sorted(image_paths), start=1):
            ans = self.analyze_image(p, question)
            results.append({"page": idx, "image_path": str(p), "answer": ans})
        return results
