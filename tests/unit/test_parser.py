import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.domain.documents.parser import DocumentParser
from src.domain.documents.exceptions import UnsupportedFileTypeError, ParsingError

@pytest.fixture
def parser():
    with patch('src.domain.documents.parser.DocumentConverter') as mock_converter:
        parser_instance = DocumentParser()
        parser_instance.converter = mock_converter.return_value
        return parser_instance

def test_validate_file_not_exists(parser):
    with pytest.raises(FileNotFoundError):
        parser._validate_file(Path("non_existent_file.pdf"))

def test_validate_unsupported_file_type(parser):
    # create a dummy file
    dummy = Path("test.invalid")
    with pytest.raises(UnsupportedFileTypeError):
        parser._validate_file(dummy)

@pytest.mark.asyncio
async def test_parse_txt_file(parser, tmp_path):
    # Create temporary txt file
    f = tmp_path / "test.txt"
    f.write_text("Hello World", encoding="utf-8")
    
    content, metadata = await parser.parse(f)
    
    assert content == "Hello World"
    assert metadata.filename == "test.txt"
    assert metadata.file_type == "txt"
    assert metadata.content_hash is not None
