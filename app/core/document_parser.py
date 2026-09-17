import io
import os
import re
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Try importing PyMuPDF (new canonical name: pymupdf, legacy alias: fitz)
try:
    try:
        import pymupdf as fitz  # preferred import (PyMuPDF >= 1.24)
    except ImportError:
        import fitz              # fallback for older installs
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) is not installed. PDF extraction will be limited.")

# Try importing winocr (Windows only)
try:
    import winocr
    WINOCR_AVAILABLE = True
except ImportError:
    WINOCR_AVAILABLE = False

# Try importing pytesseract (Cross-platform)
try:
    import pytesseract
    PYTESSERACT_MODULE_AVAILABLE = True
except ImportError:
    PYTESSERACT_MODULE_AVAILABLE = False


def find_tesseract_binary() -> Optional[str]:
    """
    Auto-detect Tesseract OCR executable across custom settings,
    system PATH, standard Linux/Render paths, and Windows installation locations.
    """
    # 1. Explicit configuration or environment variable
    custom_cmd = getattr(settings, "TESSERACT_CMD", None) or os.environ.get("TESSERACT_CMD")
    if custom_cmd:
        if os.path.isfile(custom_cmd) or shutil.which(custom_cmd):
            return custom_cmd

    # 2. System PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # 3. Common Linux / Render hosting locations
    linux_candidates = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract-ocr",
        "/opt/render/project/src/bin/tesseract",
        os.path.expanduser("~/.local/bin/tesseract"),
        os.path.join(settings.BASE_DIR, "bin", "tesseract"),
        "./bin/tesseract",
    ]
    for p in linux_candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # 4. Common Windows installation locations
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    win_candidates = [
        os.path.join(program_files, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(program_files_x86, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(local_app_data, "Programs", "Tesseract-OCR", "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in win_candidates:
        if p and os.path.isfile(p):
            return p

    return None


def get_ocr_engine_info() -> dict:
    """
    Returns diagnostic information about OCR availability,
    executable path, Tesseract version, PyMuPDF, and WinOCR support.
    """
    tesseract_path = find_tesseract_binary()
    version = None
    tesseract_available = False

    if tesseract_path and PYTESSERACT_MODULE_AVAILABLE:
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            ver_obj = pytesseract.get_tesseract_version()
            version = str(ver_obj) if ver_obj else None
            tesseract_available = True
        except Exception:
            pass

    if tesseract_path and not version:
        try:
            import subprocess
            res = subprocess.run([tesseract_path, "--version"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout:
                first_line = res.stdout.splitlines()[0].strip()
                v_match = re.search(r"tesseract\s+([0-9a-zA-Z\.\-]+)", first_line, re.IGNORECASE)
                version = v_match.group(1) if v_match else first_line
                tesseract_available = True
        except Exception:
            pass

    is_ocr_available = bool(tesseract_available or WINOCR_AVAILABLE)
    status = "ready" if is_ocr_available else ("digital_only" if PYMUPDF_AVAILABLE else "unavailable")

    return {
        "status": status,
        "ocr_available": is_ocr_available,
        "tesseract_available": tesseract_available,
        "tesseract_path": tesseract_path if (tesseract_available or tesseract_path) else None,
        "executable_path": tesseract_path if (tesseract_available or tesseract_path) else None,
        "tesseract_version": version,
        "version": version,
        "pymupdf_available": PYMUPDF_AVAILABLE,
        "winocr_available": WINOCR_AVAILABLE,
    }



class DocumentParser:
    """
    Extracts text content from uploaded PDF reports and photo/image copies of FIRs.
    Implements a robust two-pass pipeline:
      Pass 1: Direct digital text extraction via PyMuPDF (fast, 100% accurate)
      Pass 2: OCR fallback (Tesseract / WinOCR) for scanned or image-only pages.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    PDF_EXTENSIONS = {".pdf"}
    TXT_EXTENSIONS = {".txt", ".text", ".log"}

    def __init__(self):
        self.winocr_available = WINOCR_AVAILABLE
        self.pymupdf_available = PYMUPDF_AVAILABLE
        self.pytesseract_available = False
        self.tesseract_cmd: Optional[str] = None

        ocr_enabled = getattr(settings, "OCR_ENABLED", True)
        if PYTESSERACT_MODULE_AVAILABLE and ocr_enabled:
            detected_cmd = find_tesseract_binary()
            if detected_cmd:
                try:
                    pytesseract.pytesseract.tesseract_cmd = detected_cmd
                    self.pytesseract_available = True
                    self.tesseract_cmd = detected_cmd
                    logger.info(f"Tesseract OCR initialized successfully at '{detected_cmd}'")
                except Exception as e:
                    logger.warning(f"Could not configure Tesseract at '{detected_cmd}': {e}")
            else:
                logger.info(
                    "Tesseract executable not found in PATH or standard system locations. "
                    "Digital PDF extraction active; OCR will use winocr (if available) or report unavailability."
                )

    @property
    def has_ocr_engine(self) -> bool:
        """Returns True if at least one OCR engine (Tesseract or WinOCR) is usable."""
        return self.winocr_available or self.pytesseract_available

    def normalize_ocr_text(self, text: str) -> str:
        """
        Clean and normalize raw OCR output to maximize NLP entity and relationship extraction accuracy.
        Preserves Indian phone formats, vehicle registrations, bank account numbers, currency, and FIR IDs.
        """
        if not text:
            return ""

        # 1. Normalize line breaks and remove null / unprintable control chars
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 2. Fix hyphenated line-breaks (e.g. "investi- \n gation" -> "investigation")
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # 3. Normalize spaced case IDs (e.g. "FIR - 2024 - 311" -> "FIR-2024-311")
        text = re.sub(
            r"\bFIR\s*[-–—]\s*(\d{4})\s*[-–—]\s*([A-Za-z0-9]+)\b",
            r"FIR-\1-\2",
            text,
            flags=re.IGNORECASE,
        )

        # 4. Normalize spaced phone prefix (e.g. "+ 91 9811..." -> "+91 9811...")
        text = re.sub(r"\+\s*91\s*", "+91 ", text)

        # 5. Fix excessive blank lines and whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """Main async entry point to extract text from an uploaded document, text file, or image file."""
        ext = Path(filename).suffix.lower()

        if ext in self.TXT_EXTENSIONS:
            return self.extract_text_from_txt(file_bytes)
        elif ext in self.PDF_EXTENSIONS:
            return await self.extract_text_from_pdf(file_bytes)
        elif ext in self.IMAGE_EXTENSIONS:
            return await self.extract_text_from_image(file_bytes)
        else:
            supported = ", ".join(sorted(self.TXT_EXTENSIONS | self.PDF_EXTENSIONS | self.IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats are: {supported}")

    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """Extract text from plain text file bytes with encoding fallbacks."""
        if not file_bytes:
            return ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
            try:
                return file_bytes.decode(encoding).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        return file_bytes.decode("utf-8", errors="replace").strip()

    async def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from PDF using two-pass architecture:
        Pass 1: Direct digital text extraction via PyMuPDF (fast, 100% accurate)
        Pass 2: OCR fallback for scanned/image pages if OCR engine is available.
        """
        if not self.pymupdf_available:
            raise RuntimeError("PyMuPDF (fitz) is not available for PDF extraction.")

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable PDF document: {e}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return ""

        page_texts: dict[int, str] = {}
        scanned_page_nums: list[int] = []

        # Pass 1: Direct digital text extraction from PDF pages
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if len(text) > 25:
                page_texts[page_num] = text
            else:
                if text:
                    page_texts[page_num] = text
                scanned_page_nums.append(page_num)

        # Pass 2: OCR for scanned or image-only pages
        if scanned_page_nums:
            if self.has_ocr_engine:
                logger.info(f"PDF has {len(scanned_page_nums)} scanned/image page(s). Running OCR engine...")
                for page_num in scanned_page_nums:
                    try:
                        page = doc[page_num]
                        # Render page at 200 DPI for high OCR recognition accuracy
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_raw = await self._run_ocr_on_pil_image(img)
                        ocr_cleaned = self.normalize_ocr_text(ocr_raw)
                        if ocr_cleaned:
                            page_texts[page_num] = ocr_cleaned
                    except Exception as ocr_err:
                        logger.warning(f"OCR failed on page {page_num + 1}: {ocr_err}")
            else:
                logger.info(
                    f"PDF has {len(scanned_page_nums)} scanned/image page(s), but no OCR engine is available. "
                    "Using direct digital text."
                )

        doc.close()

        # Assemble extracted pages in chronological order
        extracted_pages = [page_texts[p] for p in sorted(page_texts.keys()) if page_texts[p].strip()]
        combined_text = "\n\n".join(extracted_pages).strip()

        # If no readable text exists after both passes, check engine availability
        if not combined_text:
            if not self.has_ocr_engine:
                raise RuntimeError(
                    "The uploaded PDF contains scanned images or no readable digital text, and no OCR engine "
                    "(Tesseract) is configured on the server. Please install/configure Tesseract OCR or upload a text-based document."
                )

        return combined_text

    async def extract_text_from_image(self, file_bytes: bytes) -> str:
        """Extract text from photo or scanned image using OCR."""
        if not self.has_ocr_engine:
            raise RuntimeError(
                "Cannot process image document: No OCR engine (Tesseract/WinOCR) is available on the server. "
                "Please configure Tesseract OCR to enable photo/image document ingestion."
            )

        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to read image file: {e}")

        raw_text = await self._run_ocr_on_pil_image(image)
        return self.normalize_ocr_text(raw_text)

    async def _run_ocr_on_pil_image(self, image: Image.Image) -> str:
        """Execute OCR on a PIL image using native Windows OCR or Tesseract with fallbacks."""
        # 1. Native Windows Media OCR (high performance on Windows 10/11)
        if self.winocr_available:
            try:
                max_dim = 2500
                if max(image.size) > max_dim:
                    scale = max_dim / max(image.size)
                    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                result = await winocr.recognize_pil(image, lang="en-US")
                if result and result.text:
                    return result.text
            except Exception as e:
                logger.warning(f"winocr recognition failed: {e}. Checking Tesseract fallback...")

        # 2. Tesseract OCR (Linux, Render standard, and cross-platform fallback)
        if self.pytesseract_available:
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, pytesseract.image_to_string, image)
                if text and text.strip():
                    return text
            except Exception as e:
                logger.warning(f"Tesseract OCR recognition failed: {e}")

        if not self.has_ocr_engine:
            raise RuntimeError(
                "No OCR engine available. Please install 'tesseract' on Render/Linux or configure Tesseract."
            )

        return ""
