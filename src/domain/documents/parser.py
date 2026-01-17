import hashlib
from pathlib import Path
from typing import Tuple

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except ImportError:
    # Fallback for dev environments where docling might not be installed immediately (heavy dep)
    # In production this should hard fail.
    DocumentConverter = None # type: ignore
    PdfFormatOption = None # type: ignore
    PdfPipelineOptions = None # type: ignore
    InputFormat = None # type: ignore

from src.domain.documents.models import DocumentMetadata
from src.domain.documents.exceptions import UnsupportedFileTypeError, ParsingError

class DocumentParser:
    """
    Handles parsing of documents (PDF, DOCX, TXT) into Markdown using Docling.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    def __init__(self):
        if DocumentConverter:
            # Configure to disable OCR to prevent memory leaks/crashes on small devices
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True # Keep table structure if possible
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            self.converter = None

    def _validate_file(self, file_path: Path) -> None:
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"File type {file_path.suffix} is not supported. Supported: {self.SUPPORTED_EXTENSIONS}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _calculate_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def parse(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """
        Parses a file and returns its content (Markdown) and metadata.
        """
        self._validate_file(file_path)
        
        # 1. Compute Hash
        content_hash = self._calculate_hash(file_path)
        
        # 2. Parse Content
        try:
            if file_path.suffix.lower() == '.txt':
                # Fast path for TXT
                content = file_path.read_text(encoding='utf-8')
                page_count = 1
            else:
                if not self.converter:
                     raise ParsingError("Docling is not installed or failed to initialize.")
                
                # Docling conversion
                # Note: conversion is CPU bound, in a real async app we might run this in a threadpool
                result = self.converter.convert(file_path)
                content = result.document.export_to_markdown()
                # Estimation of pages, Docling might provide this in metadata
                page_count = 0 # TODO: extract from docling result if available
                
        except Exception as e:
            raise ParsingError(f"Failed to parse document: {str(e)}") from e

        # 3. Construct Metadata
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_type=file_path.suffix.lstrip('.').lower(), # type: ignore
            page_count=page_count,
            content_hash=content_hash
        )

        return content, metadata
