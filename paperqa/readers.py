import json
import re
import traceback
from pathlib import Path
from typing import List

import fitz
from html2text import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter

from .types import Doc, Text
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast, Tuple, Any



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
                if page.find_tables():
                    pdf_texts.append(
                        Text(text=text, name=f"{doc.docname} page {i+1}", doc=doc, page_text=page_text, is_table=True))
                else:
                    pdf_texts.append(
                        Text(text=text, name=f"{doc.docname} page {i+1}", doc=doc))


        if last_text != "":
            pdf_texts.append(
                    Text(text=last_text, name=f"{doc.docname} page {fitz_file.page_count}", doc=doc))

        fitz_file.close()

        return pdf_texts
    except Exception as e:
        print(f"Error in parse_pdf_fitz: {e}")
        traceback.print_exc()


def parse_pdf(path: Path, doc: Doc, chunk_chars: int,
              overlap: int, text_splitter: TextSplitter=None) -> List[Text]:
    import pypdf

    print("the other code "*5)
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

    import os
    end_path = os.path.basename(path)
    is_table = 'Y' in end_path
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
    texts = []
    for i, t in enumerate(raw_texts):
        if(is_table is True):
            texts.append(Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc, page_text=text, is_table=True))
        else:
            texts.append(Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc, is_table=False))
    # texts = [
    #     Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc)
    #     for i, t in enumerate(raw_texts)
    # ]
    return texts

def parse_json(
    path: Path, doc: Doc, chunk_chars: int, overlap: int,
    text_splitter: TextSplitter=None,
) -> List[Text]:
    if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_chars, chunk_overlap=overlap,
                length_function=len, is_separator_regex=False,
            )
    try:
        with open(path) as f:
            file_contents = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            file_contents = f.read()

    json_contents = json.loads(file_contents)
    texts = []
    if("is_pdf" in json_contents):
        if dockey is None:
            dockey = md5sum(path)
        filename = Path(doc).name
        doc = Doc(docname=filename, citation=filename, dockey=dockey)

        is_table = True if file_contents.get('is_table') == 'True' else False
        page_text = file_contents.get('page_text')
        page_no = file_contents.get('page_no')
        page_text = page_text.encode("ascii", "ignore").decode()

        # texts = []
        for text in text_splitter.split_text(page_text):
            texts.append({
                "page": path, "text_len": len(text),
                "chunk": text, "vector_id": str(uuid.uuid4()),
                "tokens": text_splitter.count_tokens(text=text),
                "page_text": page_text,
                "is_table": is_table, "docname": docname,
                "categories": categories,
            })

    else:
        text = json_contents['text']
        doc_name = json_contents['url']

    # if text_splitter is None:
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_chars, chunk_overlap=overlap,
    #         length_function=len, is_separator_regex=False,
    #     )

        raw_texts = text_splitter.split_text(text)
        texts = [
            Text(text=t, name=f"{doc_name}", doc=doc)
            for i, t in enumerate(raw_texts)
        ]

    return texts

def parse_csv(path:Path, doc:Doc, chunk_chars:int, overlap:int, text_splitter:TextSplitter=None) -> List[Text]:
    with open(path, "r") as f:
        csv_file_data = f.read()


    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars, chunk_overlap=overlap,
            length_function=len, is_separator_regex=False,
        )

    csv_data = csv_file_data.encode("ascii", "ignore")
    csv_data = csv_data.decode()
    page_texts = text_splitter.split_text(csv_data)

    csv_texts : List[Text] = []

    for text in page_texts:
        csv_texts.append(Text(text=text,csv_text=csv_file_data, name=f"{doc.docname}", doc=doc))

    return csv_texts


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


# def unstructured_process_output(
#         self,
#         path: Path,
#         citation: Optional[str] = None,
#         docname: Optional[str] = None,
#         disable_check: bool = False,
#         dockey: Optional[DocKey] = None,
#         chunk_chars: int = 3000,
#         overlap=100,
#         text_splitter: TextSplitter = None,
#         base_dir: Path = None,
#         categories: Optional[List[str]] = None,
#     ) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:

#         if dockey is None:
#             dockey = md5sum(path)

#         # get all the files in the brase_dir
#         page_doc_list = []
#         page_doc_list.extend(
#             glob.glob(os.path.join(str(base_dir) + "**/" + "*.json"), recursive=True)
#         )

#         texts_all_pages = []
#         for idx, page_doc in enumerate(page_doc_list):
#             try:
#                 with open(page_doc) as f:
#                     file_contents = json.loads(f.read())
#             except UnicodeDecodeError:
#                 with open(page_doc, encoding="utf-8", errors="ignore") as f:
#                     file_contents = json.loads(f.read())

#             try:
#                 # docname = self._get_unique_name(docname)
#                 self.docnames.add(docname)
#                 doc = Doc(docname=docname, citation=citation, dockey=dockey)

#                 is_table = True if file_contents.get('is_table') == 'True' else False
#                 page_text = file_contents.get('page_text')
#                 page_no = file_contents.get('page_no')
#                 page_text = page_text.encode("ascii", "ignore").decode()

#                 texts = []
#                 for text in text_splitter.split_text(page_text):
#                     texts.append({
#                         "page": path, "text_len": len(text),
#                         "chunk": text, "vector_id": str(uuid.uuid4()),
#                         "tokens": text_splitter.count_tokens(text=text),
#                         "page_text": page_text,
#                         "is_table": is_table, "docname": docname,
#                         "categories": categories,
#                     })

#                 texts_all_pages += texts
#                 page_chunks_dir = base_dir / f"chunks_{page_no}"
#                 page_chunks_dir.mkdir(parents=True, exist_ok=True)
#                 chunks_file = page_chunks_dir / "text_chunks.json"

#                 with open(chunks_file, 'w') as f:
#                     json.dump(texts, f, indent=4)

#             except Exception as e:
#                 print(f"Error in unstructured_process_output: {e}")
#                 traceback.print_exc()

#         return texts_all_pages


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

    elif str_path.endswith(".json") and "meta_data.json":
        return parse_json(path, doc, chunk_chars, overlap, text_splitter)
    elif str_path.endswith(".csv"):
        return parse_csv(path, doc, chunk_chars, overlap, text_splitter)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap, text_splitter)
