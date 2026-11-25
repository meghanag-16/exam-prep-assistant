"""Convert legacy .ppt files to .pptx.

Strategy, in order:
1. LibreOffice (`soffice`) if it is on PATH. Installed in the Docker image
   (Hugging Face Space) and in the Kaggle notebook via apt.
2. Microsoft PowerPoint through COM automation, on Windows only, if pywin32
   is installed (`uv sync --extra windows-ppt`).
3. Otherwise raise PptConversionError telling the user to re-save as .pptx.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class PptConversionError(RuntimeError):
    pass


def available_converter() -> str | None:
    if shutil.which("soffice") or shutil.which("libreoffice"):
        return "libreoffice"
    if sys.platform == "win32":
        try:
            import win32com.client  # noqa: F401

            return "powerpoint"
        except ImportError:
            return None
    return None


def ppt_to_pptx(src: Path, out_dir: Path) -> Path:
    src = Path(src).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (src.stem + ".pptx")

    converter = available_converter()
    if converter == "libreoffice":
        exe = shutil.which("soffice") or shutil.which("libreoffice")
        subprocess.run(
            [exe, "--headless", "--convert-to", "pptx", "--outdir", str(out_dir), str(src)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    elif converter == "powerpoint":
        import win32com.client

        app = win32com.client.Dispatch("PowerPoint.Application")
        try:
            pres = app.Presentations.Open(str(src), WithWindow=False)
            pres.SaveAs(str(target), 24)  # 24 = ppSaveAsOpenXMLPresentation
            pres.Close()
        finally:
            app.Quit()
    else:
        raise PptConversionError(
            f"{src.name} is an old .ppt file and no converter is available here. "
            "Open it in PowerPoint and use File > Save As > .pptx, then upload again."
        )

    if not target.exists():
        raise PptConversionError(f"Conversion of {src.name} produced no .pptx file")
    return target
