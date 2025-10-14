# Point d'entrée principal du projet (exécute le pipeline complet depuis la ligne de commande).
# Point d'entrée principal du projet (exécute le pipeline complet depuis la ligne de commande).

from __future__ import annotations
import argparse
from pathlib import Path
import json
from datetime import datetime
import os

import torch
from PIL import Image
import pymupdf  # alias: fitz

# (Optionnel) éviter l'usage des fast tokenizers sur Windows anciennes configs
os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")

from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
#   1) Conversion PDF -> images (via PyMuPDF)
# ============================================================

def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 192):
    """
    Convertit chaque page du PDF en PNG.
    - dpi: 192 est un bon compromis qualité/poids (72 * 2.666...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    png_paths = []

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_p{i:03d}.png"
        pix.save(out_path.as_posix())
        png_paths.append(out_path)

    doc.close()
    return png_paths


# ============================================================
#   2) Analyse des images avec InternVL 3.5 (Transformers)
# ============================================================

def analyze_images(model_id: str, image_paths, question: str):
    """
    Analyse chaque image avec le modèle InternVL 3.5 (checkpoint -HF),
    en utilisant le processor et le chat template natifs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"   🔧 Device: {device}, dtype: {dtype}")

    # Charger processor et modèle (pas besoin de tokenizer manuel ni de patchs)
    print("   📦 Chargement du processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("   📦 Chargement du modèle...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Déplacer le modèle sur le bon device
    model = model.to(device)
    model.eval()
    print(f"   ✅ Modèle chargé avec succès sur {device}\n")

    results = []
    total = len(image_paths)

    for i, img_path in enumerate(sorted(image_paths), start=1):
        print(f"   📄 Traitement page {i}/{total}...", end=" ", flush=True)

        # Charger l'image
        image = Image.open(img_path).convert("RGB")

        # Messages format chat (le processor gère la mise en forme + les balises image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # 1) Construire le prompt textuel via le template
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # 2) Préparer les tenseurs pour le modèle (texte + image)
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # 3) Déplacer les inputs sur le bon device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 4) Génération
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # 5) Ne décoder que la partie générée (après le prompt)
        gen_only = output_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()

        results.append({"page": i, "answer": text})
        print("✅")

    return results


# ============================================================
#   3) Pipeline complet (CLI)
# ============================================================

def main():
    """Orchestration du pipeline: PDF -> Images -> Analyse -> Sauvegarde"""

    parser = argparse.ArgumentParser(
        description="Analyse de PDF avec InternVL 3.5 (Transformers + PyMuPDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  py src/main.py
  py src/main.py --pdf data/input_pdfs/mon_document.pdf
  py src/main.py --pdf rapport.pdf --dpi 300 --question "Résume cette page"
        """
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default="data/input_pdfs/evolution_pib_france.pdf",
        help="Chemin vers le fichier PDF à analyser"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=192,
        help="Résolution de conversion des pages (default: 192)"
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/page_images",
        help="Dossier de sortie pour les images"
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="data/outputs",
        help="Dossier de sortie pour les résultats"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3_5-1B-HF",   # important: checkpoint -HF
        help="Modèle HuggingFace à utiliser"
    )

    parser.add_argument(
        "--question",
        type=str,
        default=(
            "Analyse le contenu de cette page : résume le texte principal, "
            "décris les graphiques ou tableaux présents, et cite 3 informations "
            "clés avec leurs valeurs numériques."
        ),
        help="Question/prompt à poser au modèle pour chaque page"
    )

    args = parser.parse_args()

    # Préparation des chemins
    pdf_path = Path(args.pdf)
    images_dir = Path(args.images_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier l'existence du PDF
    if not pdf_path.exists():
        print(f"❌ Erreur : Le fichier {pdf_path} n'existe pas !")
        return

    print("\n" + "=" * 70)
    print("🚀 DÉMARRAGE DE L'ANALYSE PDF")
    print("=" * 70)
    print(f"📁 Fichier : {pdf_path.name}")
    print(f"🤖 Modèle  : {args.model}")
    print(f"📊 DPI     : {args.dpi}")
    print("=" * 70 + "\n")

    # Étape 1 : Conversion PDF -> Images
    print("📘 ÉTAPE 1/3 : Conversion du PDF en images...")
    try:
        images = pdf_to_images(pdf_path, images_dir, dpi=args.dpi)
        print(f"✅ {len(images)} pages converties avec succès\n")
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {e}")
        return

    # Étape 2 : Analyse
    print("🤖 ÉTAPE 2/3 : Analyse des pages avec le modèle Vision AI...")
    try:
        results = analyze_images(args.model, images, args.question)
        print(f"✅ Analyse terminée : {len(results)} pages traitées\n")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse : {e}")
        print("💡 Vérifie que tu utilises bien :")
        print("   - transformers >= 4.45 (idéalement 4.48+)")
        print("   - le checkpoint '-HF' (ex: OpenGVLab/InternVL3_5-1B-HF)")
        print("   - pillow & torchvision installés")
        print("   Commande suggérée :")
        print('   py -m pip install -U "transformers>=4.48.0" "accelerate>=0.34.0" "safetensors>=0.4.5" pillow torchvision')
        return

    # Étape 3 : Sauvegarde des résultats
    print("💾 ÉTAPE 3/3 : Sauvegarde des résultats...")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outputs_dir / f"{pdf_path.stem}_analysis_{stamp}.json"
    txt_path = outputs_dir / f"{pdf_path.stem}_analysis_{stamp}.txt"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TXT lisible
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"ANALYSE DU PDF : {pdf_path.name}\n")
        f.write(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Modèle : {args.model}\n")
        f.write(f"Question : {args.question}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"{'=' * 70}\n")
            f.write(f"PAGE {r['page']}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"{r['answer']}\n\n")

    print("✅ Résultats sauvegardés :")
    print(f"   📄 JSON : {json_path}")
    print(f"   📝 TXT  : {txt_path}")

    print("\n" + "=" * 70)
    print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS !")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
