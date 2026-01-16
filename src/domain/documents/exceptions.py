class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when the file type is not supported."""
    pass

class ParsingError(DocumentProcessingError):
    """Raised when the document parser fails."""
    pass
