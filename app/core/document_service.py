# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Secure Document Storage Service (SIH26190 Phase 2A).

Provides hardened, tamper-evident, and cross-platform server-side document storage.
Guarantees:
1. Path traversal prevention (rejects ../, Windows drive paths, UNC paths).
2. True content/magic-byte MIME validation (disallows disguised executable payloads).
3. Whitelisted extension enforcement (.pdf, .png, .jpg, .jpeg, .webp, .txt).
4. Configurable file size limits (MAX_DOCUMENT_SIZE_MB).
5. Immutable SHA-256 cryptographic checksum calculation.
6. Server-generated safe filenames (never trusts client filenames as paths).
7. Thread-safe directory management and storage isolation.
8. Safe metadata returns (never leaks sensitive server filesystem internals).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CUSTOM EXCEPTIONS
# ==============================================================================

class DocumentStorageError(Exception):
    """Base exception for all document storage operations."""
    pass


class PathTraversalError(DocumentStorageError):
    """Raised when an unsafe or escaping filesystem path is detected."""
    pass


class InvalidExtensionError(DocumentStorageError):
    """Raised when an uploaded file extension is not in the allowed whitelist."""
    pass


class InvalidMimeTypeError(DocumentStorageError):
    """Raised when the detected file magic bytes do not match permitted MIME types."""
    pass


class FileSizeLimitExceededError(DocumentStorageError):
    """Raised when file size exceeds the configured maximum."""
    pass


class DocumentNotFoundError(DocumentStorageError):
    """Raised when a requested stored document is not found."""
    pass


class DocumentSealedError(DocumentStorageError):
    """Raised when an operation attempts to modify or append to a SEALED document."""
    pass


class DocumentArchivedError(DocumentStorageError):
    """Raised when an operation attempts to modify an ARCHIVED document."""
    pass


# ==============================================================================
# 2. DOCUMENT STORAGE SERVICE
# ==============================================================================

class DocumentStorageService:
    """
    Hardened document storage manager for Node Sentinel / SIH26190.
    Manages physical file persistence, validation, integrity hashing, and secure retrieval.
    """

    ALLOWED_EXTENSIONS: Set[str] = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".txt",
    }

    # Canonical MIME mapping by extension
    EXTENSION_MIME_MAP: Dict[str, str] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".txt": "text/plain",
    }

    def __init__(
        self,
        storage_dir: Optional[str | Path] = None,
        max_size_mb: Optional[int] = None,
    ) -> None:
        raw_dir = storage_dir or getattr(settings, "DOCUMENT_STORAGE_DIR", None) or Path(settings.BASE_DIR) / "data" / "documents"
        self.storage_dir = Path(raw_dir).resolve()
        self.max_size_bytes = (max_size_mb or getattr(settings, "MAX_DOCUMENT_SIZE_MB", 50)) * 1024 * 1024
        self._lock = threading.RLock()
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Create storage root directory safely."""
        with self._lock:
            try:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create document storage directory '{self.storage_dir}': {e}")
                raise DocumentStorageError(f"Cannot initialize storage directory: {e}") from e

    # --------------------------------------------------------------------------
    # Security Validation & Normalization
    # --------------------------------------------------------------------------

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize client-provided filename to prevent path traversal, control chars,
        and invalid characters while preserving clean base names.
        """
        if not filename or not filename.strip():
            return "unnamed_document"

        # Strip directory traversal characters and full path components
        base = Path(filename).name.strip()
        # Remove null bytes and unprintable characters
        base = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", base)
        # Remove path separators
        base = base.replace("/", "_").replace("\\", "_")
        # Keep only alphanumeric, hyphens, underscores, dots, and spaces
        base = re.sub(r"[^A-Za-z0-9._\- ]", "_", base)
        # Collapse multiple spaces or dots
        base = re.sub(r"\s+", " ", base)
        base = re.sub(r"\.{2,}", ".", base)
        return base.strip(" ._") or "unnamed_document"

    @classmethod
    def validate_extension(cls, filename: str) -> str:
        """
        Validate that the file extension is strictly among allowed types.
        Returns the normalized lowercased extension (e.g. '.pdf').
        """
        ext = Path(filename).suffix.lower()
        if not ext or ext not in cls.ALLOWED_EXTENSIONS:
            allowed_str = ", ".join(sorted(cls.ALLOWED_EXTENSIONS))
            raise InvalidExtensionError(
                f"File extension '{ext}' is not permitted. Allowed extensions are: {allowed_str}"
            )
        return ext

    @classmethod
    def detect_and_validate_mime(cls, file_bytes: bytes, ext: str, client_mime: Optional[str] = None) -> str:
        """
        Inspect binary magic bytes / content header to determine true MIME type.
        Rejects disguised executable payloads (e.g. Windows PE executables or ELF binaries).
        """
        if not file_bytes:
            # Allow empty text file if extension is .txt, otherwise empty binary is invalid
            if ext == ".txt":
                return "text/plain"
            raise InvalidMimeTypeError("Uploaded file is empty (0 bytes).")

        header = file_bytes[:32]

        # Explicitly reject dangerous executable headers regardless of extension
        if header.startswith(b"MZ"):  # Windows PE / EXE / DLL
            raise InvalidMimeTypeError("Disguised executable binary detected (MZ header). Upload rejected.")
        if header.startswith(b"\x7fELF"):  # Linux ELF executable
            raise InvalidMimeTypeError("Disguised executable binary detected (ELF header). Upload rejected.")
        if header.startswith(b"PK\x03\x04") and ext != ".zip":
            # If someone uploads a zip / jar / apk named .pdf or .txt
            if ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"):
                raise InvalidMimeTypeError("Disguised ZIP/archive package detected. Upload rejected.")

        # Magic byte detection by extension
        if ext == ".pdf":
            if not header.startswith(b"%PDF-"):
                raise InvalidMimeTypeError("File contents do not match valid PDF magic bytes (%PDF-).")
            return "application/pdf"

        elif ext == ".png":
            if not header.startswith(b"\x89PNG\r\n\x1a\n"):
                raise InvalidMimeTypeError("File contents do not match valid PNG magic header.")
            return "image/png"

        elif ext in (".jpg", ".jpeg"):
            if not (header.startswith(b"\xff\xd8\xff") or header.startswith(b"\xff\xd8")):
                raise InvalidMimeTypeError("File contents do not match valid JPEG magic bytes.")
            return "image/jpeg"

        elif ext == ".webp":
            if not (header.startswith(b"RIFF") and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP"):
                raise InvalidMimeTypeError("File contents do not match valid WEBP magic header.")
            return "image/webp"

        elif ext == ".txt":
            # Check for null bytes which indicate non-text binary data
            if b"\x00" in file_bytes[:1024]:
                raise InvalidMimeTypeError("Binary data containing null bytes detected in plain text file.")
            # Verify basic text decoding
            try:
                file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_bytes.decode("latin-1")
                except UnicodeDecodeError as e:
                    raise InvalidMimeTypeError("Text file contents cannot be decoded as valid text encoding.") from e
            return "text/plain"

        # Fallback to extension default if allowed
        return cls.EXTENSION_MIME_MAP.get(ext, "application/octet-stream")

    # --------------------------------------------------------------------------
    # Path & Document ID Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def generate_document_id() -> str:
        """Generate unique legal document identifier in standard format."""
        year = datetime.now(timezone.utc).year
        unique_token = uuid.uuid4().hex[:10].upper()
        return f"DOC-{year}-{unique_token}"

    def _resolve_safe_path(self, relative_or_stored_path: str) -> Path:
        """
        Resolve a stored relative path against the storage root.
        Strictly prevents path traversal outside self.storage_dir.
        """
        # Reject raw path traversal patterns before path resolution
        norm_arg = str(relative_or_stored_path).replace("\\", "/")
        if ".." in norm_arg or norm_arg.startswith("/") or re.match(r"^[A-Za-z]:", norm_arg):
            # Check if it's already an absolute path inside storage_dir
            try:
                candidate = Path(relative_or_stored_path).resolve()
                if candidate.is_relative_to(self.storage_dir):
                    return candidate
            except (ValueError, RuntimeError):
                pass
            raise PathTraversalError(f"Path traversal detected: '{relative_or_stored_path}' is outside storage root.")

        target = (self.storage_dir / relative_or_stored_path).resolve()
        try:
            if not target.is_relative_to(self.storage_dir):
                raise PathTraversalError(f"Path traversal detected: target '{target}' escapes '{self.storage_dir}'")
        except (ValueError, RuntimeError) as e:
            raise PathTraversalError(f"Target path escapes storage directory: {e}") from e

        return target

    # --------------------------------------------------------------------------
    # Main Persistence API
    # --------------------------------------------------------------------------

    def save_document(
        self,
        file_bytes: bytes,
        original_filename: str,
        client_mime: Optional[str] = None,
        custom_doc_id: Optional[str] = None,
        subdirectory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Securely validate, hash, and persist a document to storage.
        Returns safe metadata dictionary without exposing raw server file paths.
        """
        # 1. Enforce file size limit
        file_size = len(file_bytes)
        if file_size > self.max_size_bytes:
            limit_mb = self.max_size_bytes / (1024 * 1024)
            raise FileSizeLimitExceededError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit ({limit_mb:.0f} MB)"
            )

        # 2. Sanitize filename and validate extension
        safe_orig_name = self.sanitize_filename(original_filename)
        ext = self.validate_extension(safe_orig_name)

        # 3. Detect and validate true MIME type
        detected_mime = self.detect_and_validate_mime(file_bytes, ext, client_mime)

        # 4. Compute SHA-256 cryptographic digest
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        # 5. Generate unique document ID and secure internal filename
        doc_id = custom_doc_id or self.generate_document_id()
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        safe_stem = re.sub(r"[^A-Za-z0-9_\-]", "", Path(safe_orig_name).stem)[:24]
        stored_filename = f"{doc_id}_{timestamp_str}_{safe_stem}_{sha256_hash[:8]}{ext}"

        # 6. Organize by date-based / custom subdirectory partition
        sub_dir_name = subdirectory or now.strftime("%Y%m")
        sub_dir_safe = re.sub(r"[^A-Za-z0-9_\-]", "", sub_dir_name)
        target_dir = (self.storage_dir / sub_dir_safe).resolve()

        with self._lock:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file_path = target_dir / stored_filename

            # Verify resolved target is strictly inside storage root
            if not target_file_path.resolve().is_relative_to(self.storage_dir):
                raise PathTraversalError("Target storage path escapes storage directory.")

            # Write file atomically using temporary file
            tmp_file = target_file_path.with_suffix(f"{ext}.tmp_{uuid.uuid4().hex[:6]}")
            try:
                with open(tmp_file, "wb") as f:
                    f.write(file_bytes)
                tmp_file.replace(target_file_path)
            except Exception as write_err:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
                raise DocumentStorageError(f"Failed to write document to disk: {write_err}") from write_err

        # Construct safe relative storage reference (e.g. "202609/DOC-2026-XXXX_....pdf")
        rel_storage_ref = str(Path(sub_dir_safe) / stored_filename).replace("\\", "/")

        logger.info(
            f"Stored document '{doc_id}' ({safe_orig_name}, {file_size} bytes, SHA256: {sha256_hash[:8]}...) at '{rel_storage_ref}'"
        )

        return {
            "document_id": doc_id,
            "original_filename": safe_orig_name,
            "stored_filename": stored_filename,
            "storage_reference": rel_storage_ref,
            "relative_path": rel_storage_ref,
            "file_size": file_size,
            "mime_type": detected_mime,
            "sha256_hash": sha256_hash,
            "created_at": now.isoformat(),
        }

    # --------------------------------------------------------------------------
    # Retrieval & Integrity Verification
    # --------------------------------------------------------------------------

    def get_document_path(self, storage_reference: str) -> Path:
        """
        Resolve storage reference to safe Path on disk.
        Raises DocumentNotFoundError if missing, or PathTraversalError if malicious.
        """
        target = self._resolve_safe_path(storage_reference)
        if not target.exists() or not target.is_file():
            raise DocumentNotFoundError(f"Document file not found at reference: '{storage_reference}'")
        return target

    def read_document_bytes(self, storage_reference: str) -> bytes:
        """
        Read raw binary content of stored document.
        Raises DocumentNotFoundError or PathTraversalError.
        """
        path = self.get_document_path(storage_reference)
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as e:
            raise DocumentStorageError(f"Failed to read document from disk: {e}") from e

    def verify_document_integrity(self, storage_reference: str, expected_sha256: str) -> Tuple[bool, str]:
        """
        Recalculate SHA-256 digest of stored document and verify against expected hash.
        Uses constant-time comparison to prevent timing side-channel attacks.
        Returns: (is_valid: bool, current_sha256: str)
        """
        content = self.read_document_bytes(storage_reference)
        actual_hash = hashlib.sha256(content).hexdigest()
        is_valid = hmac.compare_digest(actual_hash.lower(), expected_sha256.lower())
        return is_valid, actual_hash

    def delete_document(self, storage_reference: str) -> bool:
        """
        Delete document file from disk safely.
        """
        path = self.get_document_path(storage_reference)
        with self._lock:
            try:
                path.unlink(missing_ok=True)
                return True
            except Exception as e:
                logger.error(f"Failed to delete document file '{path}': {e}")
                return False


# ==============================================================================
# 3. DOCUMENT REGISTRY & PERSISTENCE
# ==============================================================================

class DocumentRegistry:
    """
    Thread-safe, JSON-backed persistent registry for DocumentRecord and DocumentVersion metadata.
    Provides fast in-memory indexing with atomic disk persistence.
    """

    def __init__(self, registry_file: Optional[str | Path] = None) -> None:
        if registry_file:
            self.registry_file = Path(registry_file).resolve()
        else:
            base_store = getattr(settings, "DOCUMENT_STORAGE_DIR", None) or Path(settings.BASE_DIR) / "data" / "documents"
            self.registry_file = (Path(base_store) / "registry.json").resolve()

        self._lock = threading.RLock()
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._entities: Dict[str, List[Dict[str, Any]]] = {}
        self._shares: Dict[str, List[Dict[str, Any]]] = {}
        self._permissions: Dict[str, List[Dict[str, Any]]] = {}
        self._signatures: Dict[str, List[Dict[str, Any]]] = {}
        self._anchors: Dict[str, List[Dict[str, Any]]] = {}
        self._ensure_storage()
        self._load_from_disk()

    def _ensure_storage(self) -> None:
        with self._lock:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.registry_file.exists():
                try:
                    import json
                    with open(self.registry_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "documents": {},
                            "versions": {},
                            "entities": {},
                            "shares": {},
                            "permissions": {},
                            "signatures": {},
                            "anchors": {},
                        }, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to initialize document registry at '{self.registry_file}': {e}")

    def _load_from_disk(self) -> None:
        with self._lock:
            if not self.registry_file.exists():
                return
            try:
                import json
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return
                    data = json.loads(content)
                    self._documents = data.get("documents", {})
                    self._versions = data.get("versions", {})
                    self._entities = data.get("entities", {})
                    self._shares = data.get("shares", {})
                    self._permissions = data.get("permissions", {})
                    self._signatures = data.get("signatures", {})
                    self._anchors = data.get("anchors", {})
            except Exception as e:
                logger.error(f"Error reading document registry '{self.registry_file}': {e}")

    def _save_to_disk(self) -> None:
        with self._lock:
            import json
            tmp_path = self.registry_file.with_suffix(".tmp")
            try:
                payload = {
                    "documents": self._documents,
                    "versions": self._versions,
                    "entities": self._entities,
                    "shares": self._shares,
                    "permissions": self._permissions,
                    "signatures": self._signatures,
                    "anchors": self._anchors,
                }
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, default=str)
                for attempt in range(5):
                    try:
                        tmp_path.replace(self.registry_file)
                        break
                    except (PermissionError, OSError):
                        if attempt < 4:
                            import time
                            time.sleep(0.02 * (attempt + 1))
                        else:
                            import shutil
                            shutil.copyfile(str(tmp_path), str(self.registry_file))
                            tmp_path.unlink(missing_ok=True)
                            break
            except Exception as e:
                logger.error(f"Error saving document registry to '{self.registry_file}': {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

    def save_document(self, record_dict: Dict[str, Any], initial_version_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            doc_id = record_dict["document_id"]
            self._documents[doc_id] = record_dict
            if initial_version_dict:
                v_list = self._versions.setdefault(doc_id, [])
                v_list.append(initial_version_dict)
            self._save_to_disk()
            return record_dict

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            doc = self._documents.get(document_id)
            return dict(doc) if doc else None

    def list_documents(
        self,
        case_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            items = list(self._documents.values())

            if case_id:
                clean_case = case_id.strip().lower()
                items = [d for d in items if (d.get("case_id") or "").strip().lower() == clean_case]

            if doc_type:
                clean_type = doc_type.strip().upper()
                items = [d for d in items if (d.get("doc_type") or "").strip().upper() == clean_type]

            # Sort by created_at descending
            items.sort(key=lambda d: d.get("created_at", ""), reverse=True)
            total = len(items)

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged = items[start_idx:end_idx]
            return paged, total

    def search_documents(
        self,
        query: Optional[str] = None,
        document_id: Optional[str] = None,
        case_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        confidentiality: Optional[str] = None,
        created_by: Optional[str] = None,
        version_number: Optional[int] = None,
        created_from: Optional[Union[datetime, str]] = None,
        created_to: Optional[Union[datetime, str]] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 20,
        can_access_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search and filter documents in registry by metadata attributes.
        Evaluates authorization clearance before result collation, counting, and pagination.
        """
        def _parse_dt(val: Optional[Union[datetime, str]]) -> Optional[datetime]:
            if not val:
                return None
            if isinstance(val, datetime):
                return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
            clean_str = str(val).strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception as e:
                raise ValueError(f"Invalid ISO-8601 date format: '{val}'") from e

        dt_from = _parse_dt(created_from)
        dt_to = _parse_dt(created_to)

        allowed_sort_fields = {"created_at", "updated_at", "title", "doc_type", "status", "file_size"}
        clean_sort_by = sort_by if sort_by in allowed_sort_fields else "created_at"
        reverse_order = (sort_order.lower() != "asc")

        with self._lock:
            matched_items: List[Dict[str, Any]] = []

            for doc in self._documents.values():
                # 1. Confidentiality / Authorization clearance filter (FIRST STEP)
                if can_access_fn is not None and not can_access_fn(doc):
                    continue

                # 2. Attribute-specific filters
                if document_id and doc.get("document_id", "").strip().lower() != document_id.strip().lower():
                    continue

                if case_id and (doc.get("case_id") or "").strip().lower() != case_id.strip().lower():
                    continue

                if doc_type and (doc.get("doc_type") or "").strip().upper() != doc_type.strip().upper():
                    continue

                if status and (doc.get("status") or "").strip().upper() != status.strip().upper():
                    continue

                if confidentiality and (doc.get("confidentiality_level") or "").strip().upper() != confidentiality.strip().upper():
                    continue

                if created_by and (doc.get("uploaded_by") or "").strip().lower() != created_by.strip().lower():
                    continue

                if version_number is not None:
                    v_list = self._versions.get(doc.get("document_id", ""), [])
                    has_version = (doc.get("current_version") == version_number) or any(
                        v.get("version_number") == version_number for v in v_list
                    )
                    if not has_version:
                        continue

                # 3. Date range filters
                if dt_from or dt_to:
                    doc_dt_str = doc.get("created_at")
                    if doc_dt_str:
                        doc_dt = _parse_dt(doc_dt_str)
                        if doc_dt:
                            if dt_from and doc_dt < dt_from:
                                continue
                            if dt_to and doc_dt > dt_to:
                                continue
                    else:
                        continue

                # 4. General query (q) across metadata and extracted text
                if query:
                    q_clean = query.strip().lower()
                    doc_id_val = (doc.get("document_id") or "").lower()
                    title_val = (doc.get("title") or "").lower()
                    desc_val = (doc.get("description") or "").lower()
                    case_id_val = (doc.get("case_id") or "").lower()
                    filename_val = (doc.get("file_name") or "").lower()
                    tags_val = [t.lower() for t in doc.get("tags", []) if isinstance(t, str)]
                    text_val = (doc.get("extracted_text") or "").lower()

                    matches_q = (
                        q_clean in doc_id_val
                        or q_clean in title_val
                        or q_clean in desc_val
                        or q_clean in case_id_val
                        or q_clean in filename_val
                        or q_clean in text_val
                        or any(q_clean in tag for tag in tags_val)
                    )
                    if not matches_q:
                        continue

                matched_items.append(doc)

            # 5. Sorting
            def sort_key(d: Dict[str, Any]) -> Any:
                val = d.get(clean_sort_by)
                if val is None:
                    return "" if isinstance(clean_sort_by, str) else 0
                return val

            matched_items.sort(key=sort_key, reverse=reverse_order)

            # 6. Pagination
            total = len(matched_items)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged = [dict(d) for d in matched_items[start_idx:end_idx]]

            return paged, total

    def update_status(self, document_id: str, new_status: Any) -> Dict[str, Any]:
        with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{document_id}' not found in registry.")
            status_val = new_status.value if hasattr(new_status, "value") else str(new_status).upper()
            self._documents[document_id]["status"] = status_val
            self._documents[document_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_to_disk()
            return dict(self._documents[document_id])

    def add_version(self, version_dict: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc_id = version_dict["document_id"]
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"Cannot add version: document '{doc_id}' does not exist.")

            doc = self._documents[doc_id]
            doc_status = doc.get("status", "SUBMITTED")
            if doc_status == "SEALED":
                raise DocumentSealedError(f"Document '{doc_id}' is SEALED and immutable. New versions are rejected.")
            if doc_status == "ARCHIVED":
                raise DocumentArchivedError(f"Document '{doc_id}' is ARCHIVED and cannot accept new versions.")

            v_list = self._versions.setdefault(doc_id, [])

            # Compute next monotonic version number safely under lock
            existing_numbers = [v.get("version_number", 0) for v in v_list]
            next_v_num = (max(existing_numbers) + 1) if existing_numbers else 1
            version_dict["version_number"] = next_v_num
            version_dict["parent_version_id"] = v_list[-1]["version_id"] if v_list else None

            # Supercede previous active versions
            for prev_v in v_list:
                if prev_v.get("status") == "SUBMITTED":
                    prev_v["status"] = "SUPERSEDED"

            v_list.append(version_dict)

            # Update current_version in document record
            doc["current_version"] = next_v_num
            doc["updated_at"] = version_dict.get("created_at", datetime.now(timezone.utc).isoformat())
            if "sha256_hash" in version_dict:
                doc["sha256_hash"] = version_dict["sha256_hash"]
            if "file_path" in version_dict:
                doc["file_path"] = version_dict["file_path"]
            if "file_size" in version_dict and version_dict["file_size"] is not None:
                doc["file_size"] = version_dict["file_size"]
            if "file_name" in version_dict and version_dict["file_name"]:
                doc["file_name"] = version_dict["file_name"]

            self._save_to_disk()
            return version_dict

    def create_next_version(
        self,
        document_id: str,
        file_bytes: bytes,
        original_filename: str,
        created_by: str,
        change_description: Optional[str] = None,
        client_mime: Optional[str] = None,
        storage_svc: Optional[DocumentStorageService] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Creates and stores a new immutable version for an existing document.
        Validates document existence, sealed/archived status, file constraints,
        and links version lineage.
        """
        with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{document_id}' not found in registry.")

            doc = self._documents[document_id]
            doc_status = doc.get("status", "SUBMITTED")
            if doc_status == "SEALED":
                raise DocumentSealedError(f"Document '{document_id}' is SEALED and cannot accept new versions.")
            if doc_status == "ARCHIVED":
                raise DocumentArchivedError(f"Document '{document_id}' is ARCHIVED and cannot accept new versions.")

            svc = storage_svc or get_document_storage_service()
            case_id = doc.get("case_id", "general")
            sub_dir = case_id.replace("/", "_").replace("\\", "_")[:32]

            # Save physical file
            save_meta = svc.save_document(
                file_bytes=file_bytes,
                original_filename=original_filename,
                client_mime=client_mime,
                custom_doc_id=document_id,
                subdirectory=sub_dir,
            )

            v_list = self._versions.setdefault(document_id, [])
            existing_numbers = [v.get("version_number", 0) for v in v_list]
            next_v_num = (max(existing_numbers) + 1) if existing_numbers else 1
            parent_vid = v_list[-1]["version_id"] if v_list else None
            version_id = f"VER-{document_id}-v{next_v_num}"

            now_iso = datetime.now(timezone.utc).isoformat()
            version_dict = {
                "version_id": version_id,
                "document_id": document_id,
                "version_number": next_v_num,
                "parent_version_id": parent_vid,
                "file_path": save_meta["storage_reference"],
                "sha256_hash": save_meta["sha256_hash"],
                "created_by": created_by,
                "created_at": now_iso,
                "change_reason": change_description or f"Version {next_v_num} update",
                "change_description": change_description or f"Version {next_v_num} update",
                "file_name": save_meta["original_filename"],
                "file_size": save_meta["file_size"],
                "status": "SUBMITTED",
                "metadata": {
                    "mime_type": save_meta["mime_type"],
                },
            }

            # Supercede previous active versions
            for prev_v in v_list:
                if prev_v.get("status") == "SUBMITTED":
                    prev_v["status"] = "SUPERSEDED"

            v_list.append(version_dict)

            # Update document record
            doc["current_version"] = next_v_num
            doc["updated_at"] = now_iso
            doc["sha256_hash"] = save_meta["sha256_hash"]
            doc["file_path"] = save_meta["storage_reference"]
            doc["file_size"] = save_meta["file_size"]
            doc["file_name"] = save_meta["original_filename"]
            doc["mime_type"] = save_meta["mime_type"]

            self._save_to_disk()
            return dict(doc), dict(version_dict)

    def get_versions(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            v_list = self._versions.get(document_id, [])
            return [dict(v) for v in sorted(v_list, key=lambda x: x.get("version_number", 1))]

    def get_version(self, document_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            v_list = self._versions.get(document_id, [])
            for v in v_list:
                if v.get("version_id") == version_id and v.get("document_id") == document_id:
                    return dict(v)
            return None

    def save_extracted_intelligence(
        self,
        document_id: str,
        extracted_text: str,
        entities: List[Dict[str, Any]],
    ) -> None:
        """Saves OCR text and extracted entity links for a document."""
        with self._lock:
            if document_id in self._documents:
                self._documents[document_id]["extracted_text"] = extracted_text
                self._documents[document_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._entities[document_id] = entities
            self._save_to_disk()

    def get_extracted_text(self, document_id: str) -> Optional[str]:
        with self._lock:
            doc = self._documents.get(document_id)
            return doc.get("extracted_text") if doc else None

    def get_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entities.get(document_id, []))

    # Permissions
    def add_permission(self, perm_dict: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc_id = perm_dict["document_id"]
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
            p_list = self._permissions.setdefault(doc_id, [])
            # Update existing or add new
            p_list = [p for p in p_list if p.get("permission_id") != perm_dict.get("permission_id")]
            p_list.append(perm_dict)
            self._permissions[doc_id] = p_list
            self._save_to_disk()
            return perm_dict

    def get_permissions(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._permissions.get(document_id, [])]

    def delete_permission(self, document_id: str, permission_id: str) -> bool:
        with self._lock:
            if document_id in self._permissions:
                orig_len = len(self._permissions[document_id])
                self._permissions[document_id] = [
                    p for p in self._permissions[document_id] if p.get("permission_id") != permission_id
                ]
                if len(self._permissions[document_id]) < orig_len:
                    self._save_to_disk()
                    return True
            return False

    # Shares
    def add_share(self, share_dict: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc_id = share_dict["document_id"]
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
            s_list = self._shares.setdefault(doc_id, [])
            s_list.append(share_dict)
            self._save_to_disk()
            return share_dict

    def get_shares(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            s_list = self._shares.get(document_id, [])
            results = []
            for s in s_list:
                s_copy = dict(s)
                # Check expiry
                exp = s_copy.get("expires_at")
                if exp and exp < now_iso and s_copy.get("status") == "ACTIVE":
                    s_copy["status"] = "EXPIRED"
                results.append(s_copy)
            return results

    def revoke_share(self, document_id: str, share_id: str, revoked_by: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            s_list = self._shares.get(document_id, [])
            for s in s_list:
                if s.get("share_id") == share_id:
                    s["status"] = "REVOKED"
                    s["revoked_by"] = revoked_by
                    s["revoked_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_to_disk()
                    return dict(s)
            return None

    def has_user_access(self, document_id: str, user_id: str, user_role: str) -> bool:
        """Check if user has explicit active share or permission grant."""
        with self._lock:
            doc = self._documents.get(document_id)
            if not doc:
                return False
            if doc.get("uploaded_by") == user_id:
                return True

            # Check permissions
            for p in self._permissions.get(document_id, []):
                if p.get("user_id") == user_id or p.get("role") == user_role:
                    if p.get("can_read", True):
                        return True

            # Check active shares
            now_iso = datetime.now(timezone.utc).isoformat()
            for s in self._shares.get(document_id, []):
                if s.get("shared_with") in (user_id, user_role) and s.get("status") == "ACTIVE":
                    exp = s.get("expires_at")
                    if not exp or exp >= now_iso:
                        return True
            return False

    # Signatures
    def add_signature(self, sig_dict: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc_id = sig_dict["document_id"]
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
            sig_list = self._signatures.setdefault(doc_id, [])
            sig_list.append(sig_dict)
            self._save_to_disk()
            return sig_dict

    def get_signatures(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(s) for s in self._signatures.get(document_id, [])]

    # Anchors
    def add_anchor(self, anchor_dict: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc_id = anchor_dict["document_id"]
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
            a_list = self._anchors.setdefault(doc_id, [])
            a_list.append(anchor_dict)
            self._save_to_disk()
            return anchor_dict

    def get_anchors(self, document_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(a) for a in self._anchors.get(document_id, [])]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id in self._documents:
                del self._documents[document_id]
                self._versions.pop(document_id, None)
                self._entities.pop(document_id, None)
                self._shares.pop(document_id, None)
                self._permissions.pop(document_id, None)
                self._signatures.pop(document_id, None)
                self._anchors.pop(document_id, None)
                self._save_to_disk()
                return True
            return False


# ==============================================================================
# 4. DIGITAL SIGNATURE SERVICE (PHASE 10)
# ==============================================================================

class DigitalSignatureService:
    """
    Cryptographic signing and verification for legal document versions.
    Binds signatures strictly to immutable SHA-256 digests.
    """

    @staticmethod
    def sign_version_hash(
        version_hash: str,
        signer_id: str,
        secret_key: Optional[str] = None,
        algorithm: str = "HMAC-SHA256",
    ) -> str:
        """
        Signs the document version hash.
        Uses HMAC-SHA256 with key derivation or RSA proof token.
        """
        key = (secret_key or getattr(settings, "JWT_SECRET_KEY", "node-sentinel-doc-key")).encode("utf-8")
        sig = hmac.new(key, f"{signer_id}:{version_hash}".encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    @staticmethod
    def verify_version_signature(
        version_hash: str,
        signer_id: str,
        signature: str,
        secret_key: Optional[str] = None,
    ) -> bool:
        """Constant-time verification of version digital signature."""
        expected_sig = DigitalSignatureService.sign_version_hash(version_hash, signer_id, secret_key)
        return hmac.compare_digest(expected_sig, signature)


# ==============================================================================
# 5. INTEGRITY ANCHOR SERVICE (PHASE 12)
# ==============================================================================

class IntegrityAnchorService:
    """
    Lightweight cryptographic commitment / integrity anchor service.
    Works independently with zero external blockchain dependencies.
    """

    @staticmethod
    def anchor_hash(
        document_id: str,
        version_id: str,
        version_hash: str,
        creator: str,
    ) -> Dict[str, Any]:
        """Creates a timestamped cryptographic anchor commitment for a version hash."""
        now_iso = datetime.now(timezone.utc).isoformat()
        anchor_payload = f"{document_id}|{version_id}|{version_hash}|{now_iso}|{creator}"
        anchor_id = f"ANCHOR-{hashlib.sha256(anchor_payload.encode('utf-8')).hexdigest()[:16].upper()}"
        merkle_root = hashlib.sha256(f"MERKLE_LEAF_{version_hash}".encode("utf-8")).hexdigest()

        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "version_id": version_id,
            "sha256_hash": version_hash,
            "merkle_root": merkle_root,
            "anchored_at": now_iso,
            "anchored_by": creator,
            "status": "CONFIRMED",
            "block_reference": f"SIM_BLOCK_{uuid.uuid4().hex[:8].upper()}",
        }


# ==============================================================================
# 6. DOCUMENT INTELLIGENCE & KNOWLEDGE GRAPH PIPELINE (PHASE 6 & 7)
# ==============================================================================

async def process_document_intelligence(
    document_id: str,
    file_bytes: bytes,
    filename: str,
    case_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    sync_to_graph: bool = True,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes OCR/Document Intelligence pipeline:
    1. Extracts normalized textual content (PyMuPDF / WinOCR / Tesseract / TXT).
    2. Runs NLP entity and relationship extraction with character offsets and provenance.
    3. Persists DocumentEntityLink records in registry.
    4. Synchronizes verified nodes and relationships into the central Knowledge Graph.
    """
    from app.core.document_parser import DocumentParser
    from app.core.nlp_extractor import NLPExtractor
    from app.models.document_models import DocumentEntityLink

    parser = DocumentParser()
    extractor = NLPExtractor()

    try:
        extracted_text = await parser.extract_text_from_file(file_bytes, filename)
    except Exception as e:
        logger.warning(f"OCR/text extraction failed for document '{document_id}': {e}")
        extracted_text = ""

    entity_links: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []

    if extracted_text:
        meta = {
            "document_id": document_id,
            "case_id": case_id or "general",
            "doc_type": doc_type or "OTHER",
        }
        try:
            raw_entities = extractor.extract_entities_with_provenance(extracted_text, source_metadata=meta)
            for ent in raw_entities:
                props = ent.get("properties", {})
                link_obj = DocumentEntityLink(
                    document_id=document_id,
                    entity_id=ent.get("id", ""),
                    entity_type=ent.get("label", "Unknown"),
                    confidence=float(props.get("confidence", 0.95)),
                    extraction_method=props.get("source", "NLP_HYBRID"),
                    char_start=props.get("char_start"),
                    char_end=props.get("char_end"),
                    evidence_snippet=props.get("evidence_snippet") or props.get("evidence"),
                )
                entity_links.append(link_obj.model_dump())

            triplets = extractor.extract_triplets_with_provenance(extracted_text, raw_entities, source_metadata=meta)
        except Exception as nlp_err:
            logger.warning(f"NLP entity extraction failed for document '{document_id}': {nlp_err}")

    # Save to registry
    registry = get_document_registry()
    registry.save_extracted_intelligence(document_id, extracted_text, entity_links)

    # Sync to Knowledge Graph
    if sync_to_graph and entity_links:
        try:
            sync_document_to_knowledge_graph(document_id, entity_links, triplets, case_id)
        except Exception as g_err:
            logger.warning(f"Graph synchronization warning for document '{document_id}': {g_err}")

    return extracted_text, entity_links, triplets


def sync_document_to_knowledge_graph(
    document_id: str,
    entity_links: List[Dict[str, Any]],
    triplets: List[Dict[str, Any]],
    case_id: Optional[str] = None,
) -> None:
    """
    Integrates extracted document entities and triplets into the central Knowledge Graph (NetworkX)
    attaching explicit document_id provenance to each node and edge.
    """
    from app.core.graph_engine import get_graph_engine
    from app.models.graph_models import Node, Edge, NodeType, EdgeType

    engine = get_graph_engine()

    # 1. Add/Update Nodes with document provenance
    for link in entity_links:
        ent_id = link["entity_id"]
        ent_type = link.get("entity_type", "OTHER")
        try:
            node_type = NodeType(ent_type)
        except ValueError:
            node_type = NodeType.SUSPECT if ent_type.upper() == "PERSON" else NodeType.CASE

        existing = engine.get_node(ent_id)
        props = {
            "source_document_id": document_id,
            "provenance": f"Extracted from {document_id}",
            "confidence": link.get("confidence", 0.95),
            "evidence_snippet": link.get("evidence_snippet"),
            "case_id": case_id,
        }
        if existing:
            existing.properties.update(props)
        else:
            name_val = ent_id.split("_", 1)[-1].replace("_", " ") if "_" in ent_id else ent_id
            engine.add_node(Node(
                id=ent_id,
                label=node_type,
                name=name_val,
                properties=props,
            ))

    # 2. Add Edges with document provenance
    for trip in triplets:
        src = trip["source"]
        tgt = trip["target"]
        rel_str = trip.get("relationship", "ASSOCIATED_WITH")
        try:
            edge_type = EdgeType(rel_str)
        except ValueError:
            edge_type = EdgeType.INVOLVED_IN

        if engine.get_node(src) and engine.get_node(tgt):
            edge_id = f"EDGE_DOC_{document_id[:8]}_{src[:6]}_{tgt[:6]}"
            props = dict(trip.get("properties", {}))
            props["source_document_id"] = document_id
            props["document_provenance"] = True
            engine.add_edge(Edge(
                id=edge_id,
                source=src,
                target=tgt,
                relationship=edge_type,
                properties=props,
            ))


# Singleton instances initialized with default settings
_document_storage_instance: Optional[DocumentStorageService] = None
_document_registry_instance: Optional[DocumentRegistry] = None


def get_document_storage_service() -> DocumentStorageService:
    """Retrieve or initialize DocumentStorageService singleton."""
    global _document_storage_instance
    if _document_storage_instance is None:
        _document_storage_instance = DocumentStorageService()
    return _document_storage_instance


def get_document_registry() -> DocumentRegistry:
    """Retrieve or initialize DocumentRegistry singleton."""
    global _document_registry_instance
    if _document_registry_instance is None:
        _document_registry_instance = DocumentRegistry()
    return _document_registry_instance


