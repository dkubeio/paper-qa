import asyncio
import math
import re
import string
from typing import BinaryIO, List

import pypdf
from langchain.base_language import BaseLanguageModel

from .types import StrPath


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = r"\b({0})\b(?!\w)".format(re.escape(sname))
    if re.search(pattern, text):
        return True
    return False


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0.0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def maybe_is_pdf(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number == b"%PDF"


def maybe_is_html(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return (
        magic_number == b"<htm"
        or magic_number == b"<!DO"
        or magic_number == b"<xsl"
        or magic_number == b"<!X"
    )


def strings_similarity(s1: str, s2: str) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # break the strings into words
    ss1 = set(s1.split())
    ss2 = set(s2.split())
    # return the similarity ratio
    return len(ss1.intersection(ss2)) / len(ss1.union(ss2))


def count_pdf_pages(file_path: StrPath) -> int:
    with open(file_path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    return num_pages


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def gather_with_concurrency(n: int, *coros: List) -> List:
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def guess_is_4xx(msg: str) -> bool:
    if re.search(r"4\d\d", msg):
        return True
    return False


def get_llm_name(llm: BaseLanguageModel) -> str:
    try:
        return llm.model_name  # type: ignore
    except AttributeError:
        return llm.model  # type: ignore


def fetch_sim_score(answer: str):
    patterns = [r"Score: (\[[0-9]\])", r"Score: (\d+.*)", r"Score: (\[\[[0-9]\]\])", r"Score:(\[[0-9]\])", r"Score:(\d+.*)", r"Score:(\[\[[0-9]\]\])", r"(\d+.*)"]
    k = 0
    match = None
    sim_score = 'na' 
    
    while(k < len(patterns) and not match):
        try:
            match = re.search(patterns[k], answer)
        except:
            match = None
        k = k + 1

    if match:
        sim_score = match.group(1)
        sim_score = sim_score.split(' ')[0] if sim_score.split(' ') else sim_score
        if sim_score.endswith('.'):
            sim_score = sim_score[:-1]
        try:
            sim_score = float(sim_score.strip('[').strip(']'))
        except:
            try:
                sim_score = float(sim_score.split(':')[0].split('/')[0].strip())
            except:
                print(f"ERROR: In retrieving llm_score, {sim_score}")
                sim_score = 'na' 
    else:
        print(f"Error: Did not found match, \n{answer}\n-----")

    return sim_score
