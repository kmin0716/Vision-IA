# Module responsable de la conversion des fichiers PDF en images (via PyMuPDF).

from __future__ import annotations
from pathlib import Path
from typing import List
import pymupdf  # alias historique: fitz

def pdf_to_images(
    pdf_path: Path | str,
    out_dir: Path | str,
    dpi: int = 192,
    prefix: str | None = None
) -> List[Path]:
    """
    Convertit un PDF en une image PNG par page via PyMuPDF (AUCUN poppler).
    - dpi: 144-192 = bon compromis; 300 si besoin de micro-détails.
    - prefix: préfixe des fichiers png (par défaut: nom du pdf).
    Retourne la liste des chemins PNG (triés).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = pdf_path.stem

    doc = pymupdf.open(pdf_path)
    png_paths: List[Path] = []

    # MuPDF est basé sur 72 dpi → on applique un zoom
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    try:
        for page_index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB sans alpha
            out_path = out_dir / f"{prefix}_p{page_index:03d}.png"
            pix.save(out_path.as_posix())
            png_paths.append(out_path)
    finally:
        doc.close()

    return png_paths

if __name__ == "__main__":
    # Exécution directe (debug local)
    default_pdf = Path("data/input_pdfs/evolution_pib_france.pdf")
    out = Path("data/page_images")
    imgs = pdf_to_images(default_pdf, out, dpi=192)
    print(f"{len(imgs)} images écrites dans {out.resolve()}")
