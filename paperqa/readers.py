from pathlib import Path
from typing import List, Tuple
import os
from html2text import html2text
from langchain.text_splitter import TokenTextSplitter

from .types import Doc, Text


def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> Tuple[List[Text], dict]:
    import fitz
    import tabula

    file = fitz.open(path)
    file_name = doc.docname
    base_dir = "extracted_whole_doc/%s"%file_name
    os.makedirs(base_dir, exist_ok=True)
    images_path = "%s/images"%base_dir
    tables_path = "%s/tables"%base_dir
    chunks_path = "%s/chunks"%base_dir
    #os.mkdir(images_path)
    #os.mkdir(tables_path)
    os.mkdir(chunks_path)

    count = {}
    img_count = 0
    table_count = 0
    text_count = 0

    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    chunks_list = []
    for i in range(file.page_count):
        page = file.load_page(i)
        # extract tables
        #tables = tabula.read_pdf(path, pages=i+1, silent=True)
        #if(tables):
        #    table_count += len(tables)
        #    # note: 1 csv file can contain more than 1 tables
        #    tabula.convert_into(path, os.path.join(tables_path, "%s.csv"%(i+1)),
        #                        output_format="csv",pages=(i+1), silent=True)
        # extract images
        #img_list = page.get_images()
        #for j, img in enumerate(img_list, start=1):
        #    img_count += 1
        #    xref = img[0]
        #    #Extract image
        #    base_image = file.extract_image(xref)
        #    #Store image bytes
        #    image_bytes = base_image['image']
        #    #Store image extension
        #    image_ext = base_image['ext']
        #    #Generate image file name
        #    image_name = "%s_"%(i+1) + str(j) + '.' + image_ext
        #    with open(os.path.join(images_path, image_name) , 'wb') as image_file:
        #        image_file.write(image_bytes)
        #        image_file.close()

        split += page.get_text("text", sort=True)
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
            chunks_list.append({f"{doc.docname} pages {pg}": split[:chunk_chars]})
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
            text_count += 1
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
        text_count += 1
    import json
    with open(os.path.join(chunks_path, "chunks.json"), "w") as outfile:
        json.dump(chunks_list, outfile)
    with open(os.path.join(base_dir, "count.json"), "w") as outfile:
        count["text"] = text_count
        count["image"] = img_count
        count["table"] = table_count
        json.dump(count, outfile)
    file.close()
    return texts,count


def parse_pdf(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
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
    path: Path, doc: Doc, chunk_chars: int, overlap: int, html: bool = False
) -> Tuple[List[Text], dict]:
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    count = {}
    file_name = doc.docname
    base_dir = "extracted_whole_doc/%s"%file_name
    os.makedirs(base_dir, exist_ok=True)
    images_path = "%s/images"%base_dir
    tables_path = "%s/tables"%base_dir
    chunks_path = "%s/chunks"%base_dir
    #os.mkdir(images_path)
    #os.mkdir(tables_path)
    os.mkdir(chunks_path)

    img_count = 0
    table_count = 0
    if html:
        text = html2text(text)
    # yo, no idea why but the texts are not split correctly
    text_splitter = TokenTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]
    chunks_list = [{x.name:x.text} for x in texts]
    import json
    with open(os.path.join(chunks_path, "chunks.json"), "w") as outfile:
        json.dump(chunks_list, outfile)
    with open(os.path.join(base_dir, "count.json"), "w") as outfile:
        count["text"] = len(texts)
        count["image"] = img_count
        count["table"] = table_count
        json.dump(count, outfile)
    return texts,count


def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
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
) -> Tuple[List[Text],dict]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".html"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap)
