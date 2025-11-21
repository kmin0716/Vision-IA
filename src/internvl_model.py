# Module Qwen3-VL optimisé pour l'analyse électronique

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Installe qwen-vl-utils : pip install qwen-vl-utils")

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

class InternVLAnalyzer:
    """Analyseur VLM optimisé pour schémas électroniques."""
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        print(f"🔧 Chargement de {model_id}…")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        if self.device == "cuda":
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self.model.eval()
        print("   ✅ Modèle chargé\n")

    def analyze_image(
        self, 
        image_path: Path, 
        question: str, 
        max_new_tokens: int = 800,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Analyse une image avec contrôle anti-répétition.
        
        Args:
            image_path: Chemin de l'image
            question: Question à poser
            max_new_tokens: Longueur max de réponse
            repetition_penalty: Pénalité anti-répétition (1.0-1.5)
        """
        image = Image.open(image_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,              # Déterministe
                repetition_penalty=repetition_penalty,  # Anti-répétition
                no_repeat_ngram_size=3,       # Interdit répétition de 3-grams
            )

        # Décode uniquement la génération
        gen_only = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            [gen_only],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        return text

    def analyze_batch(
        self, 
        image_paths: Iterable[Path], 
        question: str,
        max_new_tokens: int = 800
    ) -> List[Dict[str, Any]]:
        """Analyse plusieurs images en séquence."""
        results: List[Dict[str, Any]] = []
        
        for idx, p in enumerate(sorted(image_paths), start=1):
            print(f"   📄 Page {idx}…", end=" ", flush=True)
            try:
                ans = self.analyze_image(
                    p, 
                    question, 
                    max_new_tokens=max_new_tokens
                )
                results.append({
                    "page": idx,
                    "image_path": str(p),
                    "answer": ans
                })
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                results.append({
                    "page": idx,
                    "image_path": str(p),
                    "answer": f"ERREUR: {e}"
                })
        
        return results


# ==== Test local ====
if __name__ == "__main__":
    import sys
    test_image = Path("data/page_images/circuit_p001.png")
    
    if not test_image.exists():
        print("⚠️  Image de test introuvable.")
        sys.exit(1)
    
    analyzer = InternVLAnalyzer()
    
    # Question spécialisée électronique
    q = (
        "Analyse ce schéma électronique. "
        "Identifie le type de circuit, les composants et leurs valeurs, "
        "puis calcule le gain si applicable."
    )
    
    result = analyzer.analyze_image(test_image, q)
    print("\n" + "="*60)
    print("RÉSULTAT:")
    print("="*60)
    print(result)