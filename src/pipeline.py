# Orchestration : PDF → images → Qwen3-VL → fichiers

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

from pdf_to_img import pdf_to_images
from internvl_model import InternVLAnalyzer

def run_pipeline(
    pdf_path: Path | str = Path("data/input_pdfs/evolution_pib_france.pdf"),
    images_dir: Path | str = Path("data/page_images"),
    outputs_dir: Path | str = Path("data/outputs"),
    dpi: int = 192,
    question: str = (
        "Analyse le contenu de la page (schémas + texte) : "
        "résume en 5 points les tendances, cite les unités, "
        "repère 3 valeurs clés chiffrées, et donne une phrase de conclusion."
    )
) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF -> PNG
    png_paths = pdf_to_images(pdf_path, images_dir, dpi=dpi)

    # 2) Analyse par Qwen3-VL (2B)
    analyzer = InternVLAnalyzer(model_id="Qwen/Qwen3-VL-2B-Instruct")
    results = analyzer.analyze_batch(png_paths, question)

    # 3) Sauvegardes
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdf_path.stem
    json_path = outputs_dir / f"{base_name}_analysis_{stamp}.json"
    txt_path  = outputs_dir / f"{base_name}_analysis_{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pdf": str(pdf_path),
            "dpi": dpi,
            "question": question,
            "pages": results,
            "created_at": stamp,
        }, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PDF: {pdf_path}\nDPI: {dpi}\nQuestion: {question}\n\n")
        for item in results:
            f.write(f"=== Page {item['page']} ===\n")
            f.write(item["answer"] + "\n\n")

    return {
        "images": [str(p) for p in png_paths],
        "json": str(json_path),
        "txt": str(txt_path),
        "pages_count": len(png_paths),
    }

if __name__ == "__main__":
    out = run_pipeline()
    print(f"Pages rendues: {out['pages_count']}")
    print(f"JSON: {out['json']}")
    print(f"TXT : {out['txt']}")
