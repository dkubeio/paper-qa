import json
import time
from pathlib import Path
from typing import List
import fitz
from html2text import html2text
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .types import Doc, Text
import re

def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int,
                   overlap: int, text_splitter: TextSplitter = None) -> List[Text]:
    try:
        pdf_texts: List[Text] = []
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_chars, chunk_overlap=overlap,
                length_function=len, is_separator_regex=False,
            )

        last_text = ""
        # with fitz.open(path) as fitz_file:
        fitz_file = fitz.open(path)  # type: ignore

        for i in range(fitz_file.page_count):
            page = fitz_file.load_page(i)
            page_text: str = last_text + ' ' + page.get_text("text", sort=True)

            page_text = page_text.replace(u'\xa0', u' ').encode("ascii", "ignore")
            page_text = page_text.decode()
            page_text = page_text.replace('\n', ' ').replace('\r', ' ')
            page_text = re.sub(' +', ' ', page_text)

            page_texts = text_splitter.split_text(page_text)
            last_text = page_texts[-1]
            texts = page_texts[:-1]

            # create chunks per page
            for text in texts:
                pdf_texts.append(
                    Text(text=text, name=f"{doc.docname} page {i+1}", doc=doc))

        if last_text != "":
            pdf_texts.append(
                    Text(text=last_text, name=f"{doc.docname} page {fitz_file.page_count}", doc=doc))

        fitz_file.close()

        return pdf_texts
    except Exception as e:
        print(f"Error in parse_pdf_fitz: {e}")
        import traceback
        traceback.print_exc()




def parse_pdf(path: Path, doc: Doc, chunk_chars: int,
              overlap: int, text_splitter: TextSplitter=None) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    pdfFileObj.close()
    return texts


def parse_txt(
    path: Path, doc: Doc, chunk_chars: int, overlap: int,
    html: bool = False, text_splitter: TextSplitter=None
) -> List[Text]:
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if html:
        text = html2text(text)

    text = text.encode("ascii", "ignore")
    text = text.decode()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(' +',' ', text)

    # yo, no idea why but the texts are not split correctly
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars, chunk_overlap=overlap,
            length_function=len, is_separator_regex=False,
        )

    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]
    return texts

def parse_json(
    path: Path, doc: Doc, chunk_chars: int, overlap: int,
    text_splitter: TextSplitter=None,
) -> List[Text]:
    try:
        with open(path) as f:
            file_contents = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            file_contents = f.read()

    json_contents = json.loads(file_contents)
    text = json_contents['text']
    doc_name = json_contents['url']

    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars, chunk_overlap=overlap,
            length_function=len, is_separator_regex=False,
        )

    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc_name}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]

    return texts

def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int,
                   token_splitter: TextSplitter = None
) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
            split += line
            if len(split) > chunk_chars:
                texts.append(
                    Text(
                        text=split[:chunk_chars],
                        name=f"{doc.docname} lines {last_line}-{i}",
                        doc=doc,
                    )
                )
                split = split[chunk_chars - overlap :]
                last_line = i
    if len(split) > overlap:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


def read_doc(
    path: Path,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
    text_splitter: TextSplitter = None,
) -> List[Text]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)

        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap, text_splitter)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap, text_splitter)

    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap, False, text_splitter)

    elif str_path.endswith(".html") or str_path.endswith(".htm"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True, text_splitter=text_splitter)

    elif str_path.endswith(".json") and "meta_data.json" not in str_path:
        return parse_json(path, doc, chunk_chars, overlap, text_splitter)

    else:
        return parse_code_txt(path, doc, chunk_chars, overlap, text_splitter)
