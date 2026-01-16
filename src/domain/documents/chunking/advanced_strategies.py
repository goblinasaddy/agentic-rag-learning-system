import re
from typing import List, Optional
from uuid import UUID
from src.domain.documents.chunking.base import BaseChunker, ChunkerConfig
from src.domain.documents.models import ChunkMetadata

# Try to import sentence types
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

class SemanticChunker(BaseChunker):
    def __init__(self, config: ChunkerConfig):
        if not SentenceTransformer:
            raise ImportError("sentence-transformers is required for SemanticChunking")
        # Load model (this is heavy, in prod we might dependency inject or lazy load)
        self.model = SentenceTransformer(config.embedding_model)
        self.threshold = config.breakpoint_threshold_amount / 100.0 # e.g. 0.95

    def chunk(self, text: str, doc_id: UUID) -> List[ChunkMetadata]:
        # 1. Split into sentences
        sentences = re.split(r'(?<=[.?!])\s+', text)
        if not sentences:
            return []
            
        # 2. Embed sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # 3. Calculate cosine distances
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_start_idx = 0
        
        for i in range(1, len(sentences)):
            # Distance between current sentence and previous
            score = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            
            if score >= self.threshold:
                current_chunk_sentences.append(sentences[i])
            else:
                # Breakpoint found
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(ChunkMetadata(
                    doc_id=doc_id,
                    start_char_idx=current_start_idx,
                    end_char_idx=current_start_idx + len(chunk_text),
                    page_number=None,
                    section_title=None
                ))
                current_start_idx += len(chunk_text) + 1 # +1 for space
                current_chunk_sentences = [sentences[i]]
        
        # Flush last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(ChunkMetadata(
                doc_id=doc_id,
                start_char_idx=current_start_idx,
                end_char_idx=current_start_idx + len(chunk_text),
                page_number=None,
                section_title=None
            ))
            
        return chunks

class MarkdownChunker(BaseChunker):
    def __init__(self, config: ChunkerConfig):
        # Matches #, ##, ### headers
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    def chunk(self, text: str, doc_id: UUID) -> List[ChunkMetadata]:
        # Simple splitting by headers
        # In a real implementation we would strictly follow the markdown structure.
        
        # Splits by explicit newlines before headers
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_header = None
        start_idx = 0
        
        for line in lines:
            is_header = line.strip().startswith('#')
            if is_header:
                # Flush previous
                if current_chunk_lines:
                    content = "\n".join(current_chunk_lines)
                    chunks.append(ChunkMetadata(
                        doc_id=doc_id,
                        start_char_idx=start_idx,
                        end_char_idx=start_idx + len(content),
                        section_title=current_header
                    ))
                    start_idx += len(content) + 1
                
                current_header = line.strip().replace('#', '').strip()
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
        
        # Flush last
        if current_chunk_lines:
            content = "\n".join(current_chunk_lines)
            chunks.append(ChunkMetadata(
                doc_id=doc_id,
                start_char_idx=start_idx,
                end_char_idx=start_idx + len(content),
                section_title=current_header
            ))
            
        return chunks
