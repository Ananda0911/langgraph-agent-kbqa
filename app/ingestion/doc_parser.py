"""DocParser: 多格式文档解析与分块。

支持 pdf / txt / md / docx，解析后按「段落 + 滑动窗口」策略切分为带重叠的文本块，
每块附带 doc_id / 块序号 / 字符位置等元数据。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    seq: int              # 块序号
    text: str
    start: int = 0        # 原文中的起始位置

    def meta(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "seq": self.seq,
            "start": self.start,
        }


@dataclass
class ParsedDocument:
    doc_id: str
    name: str
    text: str
    source: str = ""      # 来源文件路径
    chunks: list[Chunk] = field(default_factory=list)


def _read_text(path: Path) -> str:
    """兼容 utf-8 / gbk 的文本读取。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要 pypdf：pip install pypdf") from e
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    """零依赖 docx 解析：直接读取 word/document.xml 中的文本节点。"""
    import zipfile
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    parts = ["".join(t.text or "" for t in p.iter(f"{ns}t")) for p in root.iter(f"{ns}p")]
    return "\n".join(x for x in parts if x.strip())


def parse_file(path: str | Path) -> ParsedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _parse_pdf(path)
    elif suffix == ".docx":
        text = _parse_docx(path)
    elif suffix in (".txt", ".md", ".markdown", ""):
        text = _read_text(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}（支持 pdf/docx/txt/md）")
    doc_id = "doc_" + hashlib.md5(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()[:10]
    return ParsedDocument(doc_id=doc_id, name=path.name, text=text, source=str(path))


def _split_long_paragraph(para: str, size: int, overlap: int) -> list[str]:
    """超长段落按句子组装成若干块，块间保留 overlap 字符。"""
    sentences = [s for s in _SENT_SPLIT.split(para) if s.strip()]
    blocks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if buf_len + len(sent) > size and buf:
            blocks.append("".join(buf))
            # 保留尾部 overlap 作为下一块开头
            tail = "".join(buf)
            tail = tail[-overlap:] if overlap > 0 else ""
            buf = [tail] if tail else []
            buf_len = len(tail)
        buf.append(sent)
        buf_len += len(sent)
    if buf:
        blocks.append("".join(buf))
    return blocks or [para]


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """按段落切分，超长段按句子二次切分，附带滑动窗口重叠。"""
    chunks: list[Chunk] = []
    pos = 0
    for para in _PARA_SPLIT.split(doc.text):
        para = para.strip()
        if not para:
            pos += 1
            continue
        pieces = (
            _split_long_paragraph(para, chunk_size, overlap)
            if len(para) > chunk_size
            else [para]
        )
        for piece in pieces:
            chunk_id = hashlib.md5(f"{doc.doc_id}:{len(chunks)}:{piece[:64]}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    seq=len(chunks),
                    text=piece,
                    start=pos,
                )
            )
            pos += len(piece)
    doc.chunks = chunks
    return chunks
