# Version optimisée avec barre de progression détaillée

from __future__ import annotations
import argparse
from pathlib import Path
import json
from datetime import datetime
import os
import time

import torch
from PIL import Image
import pymupdf

os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")

from transformers import AutoProcessor, AutoModelForVision2Seq
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Installe qwen-vl-utils : pip install qwen-vl-utils")

# Pour la barre de progression
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("💡 Pour avoir une belle barre de progression : pip install tqdm")


# ============ Barre de progression personnalisée ============

class ProgressTracker:
    """Affiche la progression en temps réel."""
    
    def __init__(self, total: int, desc: str = "Traitement"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        
    def update(self, page_num: int):
        """Met à jour la progression."""
        self.current = page_num
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)
            eta_str = f"{int(remaining//60)}min {int(remaining%60)}s"
        else:
            eta_str = "calcul..."
        
        percent = (self.current / self.total) * 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.desc} [{bar}] {percent:.1f}% | "
              f"Page {self.current}/{self.total} | "
              f"Temps: {int(elapsed)}s | "
              f"ETA: {eta_str}", end="", flush=True)
    
    def finish(self):
        """Affiche la completion."""
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc} [{'█'*40}] 100% | "
              f"{self.total}/{self.total} pages | "
              f"Terminé en {int(elapsed)}s ({elapsed/self.total:.1f}s/page)")
        print()


# ============ 1) PDF -> images avec progression ============

def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 150):
    """Convertit chaque page du PDF en PNG avec barre de progression."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    png_paths = []
    
    total_pages = len(doc)
    
    if HAS_TQDM:
        iterator = tqdm(enumerate(doc, start=1), 
                       total=total_pages,
                       desc="📄 Conversion PDF",
                       unit="page",
                       colour="blue")
    else:
        print(f"📄 Conversion de {total_pages} pages...")
        iterator = enumerate(doc, start=1)
    
    for i, page in iterator:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_p{i:03d}.png"
        pix.save(out_path.as_posix())
        png_paths.append(out_path)
        
        if not HAS_TQDM:
            print(f"  ✓ Page {i}/{total_pages}")
    
    doc.close()
    return png_paths


# ============ 2) Analyse avec Qwen3-VL et progression ============

def analyze_images(model_id: str, image_paths, question: str):
    """Analyse avec barre de progression détaillée."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\n🔧 Configuration :")
    print(f"   • Device : {device}")
    print(f"   • Dtype  : {dtype}")
    print(f"   • Modèle : {model_id}")
    
    print("\n📦 Chargement du modèle...")
    load_start = time.time()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if device == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
    
    model.eval()
    load_time = time.time() - load_start
    print(f"✅ Modèle chargé en {load_time:.1f}s\n")

    results = []
    total = len(image_paths)
    
    # Barre de progression
    if HAS_TQDM:
        iterator = tqdm(enumerate(sorted(image_paths), start=1),
                       total=total,
                       desc="🤖 Analyse VLM",
                       unit="page",
                       colour="green")
    else:
        print(f"🤖 Analyse de {total} pages...\n")
        tracker = ProgressTracker(total, "🤖 Analyse")
        iterator = enumerate(sorted(image_paths), start=1)

    analysis_times = []
    
    for i, img_path in iterator:
        page_start = time.time()
        
        image = Image.open(img_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

        gen_only = out_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(
            [gen_only],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        page_time = time.time() - page_start
        analysis_times.append(page_time)
        
        results.append({
            "page": i,
            "answer": text,
            "processing_time_seconds": round(page_time, 2)
        })
        
        if not HAS_TQDM:
            tracker.update(i)
    
    if not HAS_TQDM:
        tracker.finish()
    
    # Statistiques finales
    avg_time = sum(analysis_times) / len(analysis_times)
    total_time = sum(analysis_times)
    print(f"\n📊 Statistiques :")
    print(f"   • Temps moyen/page : {avg_time:.1f}s")
    print(f"   • Temps total      : {int(total_time//60)}min {int(total_time%60)}s")
    print(f"   • Page la plus rapide : {min(analysis_times):.1f}s")
    print(f"   • Page la plus lente  : {max(analysis_times):.1f}s")
    
    return results


# ============ 3) CLI orchestration ============

def main():
    parser = argparse.ArgumentParser(
        description="Analyse PDF électronique avec barre de progression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdf", type=str, default="data/input_pdfs/aop_facile.pdf",
        help="Chemin vers le fichier PDF à analyser"
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI de rendu (150 recommandé)")
    parser.add_argument("--images_dir", type=str, default="data/page_images")
    parser.add_argument("--outputs_dir", type=str, default="data/outputs")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
        help="Modèle HuggingFace"
    )
    parser.add_argument(
        "--question", type=str,
        default=(
            "Tu es un expert en électronique. Analyse cette page d'exercices.\n\n"
            "RÈGLES IMPORTANTES :\n"
            "- N'utilise JAMAIS de notation LaTeX (pas de \\(, \\), \\frac, etc.)\n"
            "- Écris les formules en texte simple : G = -R2/R1 = -45/2.5 = -18\n"
            "- Fais TOUS les calculs numériques étape par étape\n"
            "- Donne le résultat final avec l'unité\n\n"
            "Pour chaque exercice visible :\n"
            "1. Type de circuit : [nom du circuit]\n"
            "2. Composants : R1 = [valeur], R2 = [valeur], etc.\n"
            "3. Formule : [formule en texte simple]\n"
            "4. Calcul : [étapes détaillées]\n"
            "5. Résultat final : [valeur avec unité]\n\n"
            "Exemple de bonne réponse :\n"
            "Exercice 1 - Amplificateur inverseur\n"
            "- R1 = 2.5 kΩ\n"
            "- R2 = 45 kΩ\n"
            "- Formule du gain : G = -R2/R1\n"
            "- Calcul : G = -45/2.5 = -18\n"
            "- Résultat : Le gain est de -18 (sans unité)"
        ),
        help="Question posée au modèle"
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    images_dir = Path(args.images_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"❌ Erreur : Le fichier {pdf_path} n'existe pas !")
        return

    # Header
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "🚀 ANALYSE PDF ÉLECTRONIQUE" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n📄 Fichier : {pdf_path.name}")
    print(f"🤖 Modèle  : {args.model}")
    print(f"📊 DPI     : {args.dpi}")
    print(f"⏰ Début   : {datetime.now().strftime('%H:%M:%S')}")
    print("─" * 70)

    total_start = time.time()

    # Étape 1 : PDF -> images
    print("\n" + "=" * 70)
    print("📘 ÉTAPE 1/3 : Conversion du PDF en images")
    print("=" * 70)
    try:
        images = pdf_to_images(pdf_path, images_dir, dpi=args.dpi)
        print(f"✅ {len(images)} pages converties avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {e}")
        return

    # Étape 2 : Analyse
    print("\n" + "=" * 70)
    print("🤖 ÉTAPE 2/3 : Analyse avec Qwen3-VL-2B-Instruct")
    print("=" * 70)
    try:
        results = analyze_images(args.model, images, args.question)
        print(f"\n✅ Analyse terminée : {len(results)} pages traitées")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse : {e}")
        import traceback
        traceback.print_exc()
        return

    # Étape 3 : Sauvegarde
    print("\n" + "=" * 70)
    print("💾 ÉTAPE 3/3 : Sauvegarde des résultats")
    print("=" * 70)
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outputs_dir / f"{pdf_path.stem}_analysis_{stamp}.json"
    txt_path  = outputs_dir / f"{pdf_path.stem}_analysis_{stamp}.txt"

    # JSON avec métadonnées
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pdf": pdf_path.name,
            "model": args.model,
            "dpi": args.dpi,
            "question": args.question,
            "total_processing_time_seconds": round(time.time() - total_start, 2),
            "pages": results,
            "timestamp": stamp,
        }, f, ensure_ascii=False, indent=2)

    # TXT lisible
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"ANALYSE ÉLECTRONIQUE : {pdf_path.name}\n")
        f.write(f"Modèle : {args.model}\n")
        f.write(f"Date   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Temps total : {int((time.time()-total_start)//60)}min {int((time.time()-total_start)%60)}s\n")
        f.write("=" * 70 + "\n\n")
        
        for r in results:
            f.write(f"{'=' * 70}\n")
            f.write(f"PAGE {r['page']} (traité en {r['processing_time_seconds']}s)\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(f"{r['answer']}\n\n")

    total_time = time.time() - total_start
    
    print(f"✅ Résultats sauvegardés :")
    print(f"   📄 JSON : {json_path}")
    print(f"   📝 TXT  : {txt_path}")
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 23 + "🎉 ANALYSE TERMINÉE !" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n⏱️  Temps total : {int(total_time//60)} min {int(total_time%60)} s")
    print(f"📊 Vitesse moyenne : {total_time/len(results):.1f}s par page")
    print()


if __name__ == "__main__":
    main()