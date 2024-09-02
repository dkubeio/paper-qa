import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast, Tuple, Any
import glob
import traceback
from urllib.parse import quote
import json
import re

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, validator
from sentence_transformers import CrossEncoder
from pathlib import Path

from .chains import get_score, make_chain
from .paths import PAPERQA_DIR
from .readers import read_doc
from .types import Answer, CallbackFactory, Context, Doc, DocKey, PromptCollection, Text, Faq_Text
from .utils import (
    gather_with_concurrency,
    get_llm_name,
    guess_is_4xx,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
)


class NoMatchesFoundException(Exception):
    pass


class Docs(BaseModel, arbitrary_types_allowed=True, smart_union=True):
    """A collection of documents to be used for answering questions."""

    docs: Dict[DocKey, Doc] = {}
    texts: List[Text] = []
    docnames: Set[str] = set()
    texts_index: Optional[VectorStore] = None
    doc_index: Optional[VectorStore] = None
    cache_index: Optional[VectorStore] = None
    llm: Union[str, BaseLanguageModel] = ChatOpenAI(
        temperature=0.1, model="gpt-3.5-turbo", client=None
    )
    summary_llm: Optional[Union[str, BaseLanguageModel]] = None
    name: str = "default"
    index_path: Optional[Path] = PAPERQA_DIR / name
    embeddings: Embeddings = OpenAIEmbeddings(client=None)
    max_concurrent: int = 5
    deleted_dockeys: Set[DocKey] = set()
    prompts: PromptCollection = PromptCollection()
    memory: bool = False
    memory_model: Optional[BaseChatMemory] = None
    jit_texts_index: bool = False

    # TODO: Not sure how to get this to work
    # while also passing mypy checks
    @validator("llm", "summary_llm")
    def check_llm(cls, v: Union[BaseLanguageModel, str]) -> BaseLanguageModel:
        if type(v) is str:
            return ChatOpenAI(temperature=0.1, model=v, client=None)
        return cast(BaseLanguageModel, v)

    @validator("summary_llm", always=True)
    def copy_llm_if_not_set(cls, v, values):
        return v or values["llm"]

    @validator("memory_model", always=True)
    def check_memory_model(cls, v, values):
        if values["memory"]:
            if v is None:
                return ConversationTokenBufferMemory(
                    llm=values["summary_llm"],
                    max_token_limit=512,
                    memory_key="memory",
                    human_prefix="Question",
                    ai_prefix="Answer",
                    input_key="Question",
                    output_key="Answer",
                )
            if v.memory_variables()[0] != "memory":
                raise ValueError("Memory model must have memory_variables=['memory']")
            return values["memory_model"]
        return None

    def clear_docs(self):
        self.texts = []
        self.docs = {}
        self.docnames = set()

    def update_memoryllm(self, memoryllm):
        self.memory_model = memoryllm

    def update_llm(
        self,
        llm: Union[BaseLanguageModel, str],
        summary_llm: Optional[Union[BaseLanguageModel, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0.1, model=llm, client=None)
        if type(summary_llm) is str:
            summary_llm = ChatOpenAI(temperature=0.1, model=summary_llm, client=None)
        self.llm = cast(BaseLanguageModel, llm)
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = cast(BaseLanguageModel, summary_llm)

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name"""
        suffix = ""
        while docname + suffix in self.docnames:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        docname += suffix
        return docname

    def add_file(
        self,
        file: BinaryIO,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Add a document to the collection."""
        # just put in temp file and use existing method
        suffix = ".txt"
        if maybe_is_pdf(file):
            suffix = ".pdf"
        elif maybe_is_html(file):
            suffix = ".html"

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(file.read())
            f.seek(0)
            return self.add(
                Path(f.name),
                citation=citation,
                docname=docname,
                dockey=dockey,
                chunk_chars=chunk_chars,
            )

    def add_url(
        self,
        url: str,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Tuple[Optional[str], Optional[list[str]]]:
        """Add a document to the collection."""
        import urllib.request

        with urllib.request.urlopen(url) as f:
            # need to wrap to enable seek
            file = BytesIO(f.read())
            return self.add_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                chunk_chars=chunk_chars,
            )

    def add(
        self,
        path: Path,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        disable_check: bool = False,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Add a document to the collection."""
        if dockey is None:
            dockey = md5sum(path)
        if citation is None:
            # skip system because it's too hesitant to answer
            cite_chain = make_chain(
                prompt=self.prompts.cite,
                llm=cast(BaseLanguageModel, self.summary_llm),
                skip_system=True,
            )
            # peak first chunk
            fake_doc = Doc(docname="", citation="", dockey=dockey)
            texts = read_doc(path, fake_doc, chunk_chars=chunk_chars, overlap=100)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            citation = cite_chain.run(texts[0].text)
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if docname is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)  # type: ignore
            else:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse docname from citation {citation}. "
                    "Consider just passing key explicitly - e.g. docs.py "
                    "(path, citation, key='mykey')"
                )
            year = ""
            match = re.search(r"(\d{4})", citation)
            if match is not None:
                year = match.group(1)  # type: ignore
            docname = f"{author}{year}"
        docname = self._get_unique_name(docname)
        doc = Doc(docname=docname, citation=citation, dockey=dockey)
        texts = read_doc(path, doc, chunk_chars=chunk_chars, overlap=100)
        # loose check to see if document was loaded
        if (
            len(texts) == 0
            or len(texts[0].text) < 10
            or (not disable_check and not maybe_is_text(texts[0].text))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )
        if self.add_texts(texts, doc):
            text_chunks = [x.text for x in texts]
            return docname, text_chunks
        return None, None


    def generate_chunks(
        self,
        path: Path,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        disable_check: bool = False,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
        overlap=100,
        text_splitter: TextSplitter = None,
        base_dir: Path = None,
    ) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:
        """Add a document to the collection."""
        if dockey is None:
            dockey = md5sum(path)

        if citation is None:
            # skip system because it's too hesitant to answer
            cite_chain = make_chain(
                prompt=self.prompts.cite,
                llm=cast(BaseLanguageModel, self.summary_llm),
                skip_system=True,
            )
            # peak first chunk
            fake_doc = Doc(docname="", citation="", dockey=dockey)
            texts = read_doc(path, fake_doc, chunk_chars=chunk_chars, overlap=overlap, text_splitter=text_splitter,
                             base_dir=base_dir)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            citation = cite_chain.run(texts[0].text)
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if docname is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)  # type: ignore
            else:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse docname from citation {citation}. "
                    "Consider just passing key explicitly - e.g. docs.py "
                    "(path, citation, key='mykey')"
                )
            year = ""
            match = re.search(r"(\d{4})", citation)
            if match is not None:
                year = match.group(1)  # type: ignore
            docname = f"{author}{year}"

        docname = self._get_unique_name(docname)
        self.docnames.add(docname)
        doc = Doc(docname=docname, citation=citation, dockey=dockey)
        texts = read_doc(path, doc, chunk_chars=chunk_chars, overlap=overlap, text_splitter=text_splitter)
        # loose check to see if document was loaded
        if (
            len(texts) == 0
            or len(texts[0].text) < 10
            or (not disable_check and not maybe_is_text(texts[0].text))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )

        update_texts = []
        for index, text in enumerate(texts):
            update_texts.append(text)
            if text_splitter.count_tokens(text=text.text) < 100:   
                if index > 0:
                    update_texts[index - 1].text += " "
                    update_texts[index - 1].text += text.text
                    update_texts.pop(index)

        if update_texts and Path(path).suffix == ".json":
            docname = update_texts[0].name

        text_chunks = []
        for x in update_texts:
            if x.doc.docname.endswith('.csv'):
                text_chunks.append({
                    "page": x.name, "text_len": len(x.text),
                    "chunk": x.text, "vector_id": str(uuid.uuid4()),
                    "tokens": text_splitter.count_tokens(text=x.text),
                    "csv_text": x.csv_text, "docname": docname,
                    "doc_source": x.doc_source,
                    "state_category": x.state_category,
                })
            else:
                text_chunks.append({
                    "page": x.name, "text_len": len(x.text),
                    "chunk": x.text, "vector_id": str(uuid.uuid4()),
                    "tokens": text_splitter.count_tokens(text=x.text),
                    "page_text": x.page_text, "page_no" : x.page_no,
                    "is_table": x.is_table, "docname": docname,
                    "ext_path": x.ext_path,
                    "doc_source": x.doc_source,
                    "state_category": x.state_category,
                })

        return docname, text_chunks

    def add_texts(
        self,
        texts: List[Text],
        doc: Doc,
        is_csv: Optional[bool] = None,
        sllm_qna: Optional[bool] = False,
    ) -> bool:
        """Add chunked texts to the collection. This is useful if you have already chunked the texts yourself.

        Returns True if the document was added, False if it was already in the collection.
        """
        if doc.dockey in self.docs:
            return False

        if len(texts) == 0:
            raise ValueError("No texts to add.")

        if doc.docname in self.docnames:
            new_docname = self._get_unique_name(doc.docname)
            if not sllm_qna:
                for t in texts:
                    t.name = t.name.replace(doc.docname, new_docname)
            doc.docname = new_docname

        if texts[0].embeddings is None:
            if sllm_qna:
                text_embeddings = self.embeddings.embed_documents([t.question for t in texts])
            else:
                text_embeddings = self.embeddings.embed_documents([t.text for t in texts])
            for i, t in enumerate(texts):
                t.embeddings = text_embeddings[i]
        else:
            text_embeddings = cast(List[List[float]], [t.embeddings for t in texts])


        vector_ids = [x.vector_id for x in texts]
        if self.texts_index is not None and not sllm_qna:
            try:
                # TODO: Simplify - super weird
                if is_csv == True:
                    vec_store_text_and_embeddings = list(
                        map(lambda x: (x.csv_text, x.embeddings), texts)
                    )
                else:
                    vec_store_text_and_embeddings = list(
                        map(lambda x: (x.text, x.embeddings), texts)
                    )

                
                self.texts_index.add_embeddings(  # type: ignore
                    vec_store_text_and_embeddings,
                    ids=vector_ids,
                    metadatas=[t.dict(exclude={"embeddings", "text"}) for t in texts],
                )

                self.texts += texts
            except AttributeError:
                raise ValueError("Need a vector store that supports adding embeddings.")

        if self.cache_index is not None and sllm_qna:
            try:
                vec_store_text_and_embeddings = list(
                    map(lambda x: (x.answer, x.embeddings), texts)
                )
                
                self.cache_index.add_embeddings(
                    vec_store_text_and_embeddings,
                    ids=vector_ids,
                    metadatas=[t.dict(exclude={"embeddings", "answer"}) for t in texts],
                )
            
                self.texts += texts
            except AttributeError:
                raise ValueError("Need a vector store that supports adding faq embeddings")

        if not sllm_qna:
            if self.doc_index is not None:
                #self.doc_index.add_texts([doc.citation], metadatas=[doc.dict()])
                self.doc_index.add_texts(texts=[json.dumps(doc, default=vars)], metadatas=[doc.dict()])

            self.docs[doc.dockey] = doc
            if self.texts_index is None:
                self.texts += texts

            self.docnames.add(doc.docname)

        return True

    def delete(
        self, name: Optional[str] = None, dockey: Optional[DocKey] = None
    ) -> None:
        """Delete a document from the collection."""
        if not name.startswith('http'):
            name = os.path.basename(name)
        doc_list = []
        dockey_list = []
        if name is not None:
            # doc = next((doc for doc in self.docs.values() if doc.docname.split(' ')[:-2] == name), None)
            for doc in self.docs.values():
                if name in doc.docname:
                    doc_list.append(doc)
                    dockey_list.append(doc.dockey)
            if doc is []:
                return
            for doc in doc_list:
                self.docnames.remove(doc.docname)
            # dockey = doc.dockey
        if dockey is []:
            return

        if self.doc_index is not None:
            for dockey in dockey_list:
                # Delete docs with the dockey attribute
                self.doc_index.delete_by_attribute({'dockey':dockey})
                del self.docs[dockey]
                self.deleted_dockeys.add(dockey)
            # Delete all texts with the name attribute
        if self.texts_index is not None:
            for doc in doc_list:
                self.texts_index.delete_by_attribute({'name':doc.docname})


    async def adoc_match(
        self,
        query: str,
        k: int = 25,
        rerank: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Set[DocKey]:
        """Return a list of dockeys that match the query."""
        if self.doc_index is None:
            if len(self.docs) == 0:
                return set()
            texts = [doc.citation for doc in self.docs.values()]
            metadatas = [d.dict() for d in self.docs.values()]
            self.doc_index = FAISS.from_texts(
                texts, metadatas=metadatas, embedding=self.embeddings
            )
        matches = self.doc_index.max_marginal_relevance_search(
            query, k=k + len(self.deleted_dockeys)
        )

        # temporary work for fm-controller crash. Need to be removed.
        # Need to update the doc index when ever it changes in the dataset
        # but for now we are doing it here.
        matches = []
        # filter the matches
        matches = [
            m for m in matches if m.metadata["dockey"] not in self.deleted_dockeys
        ]
        try:
            # for backwards compatibility (old pickled objects)
            matched_docs = [self.docs[m.metadata["dockey"]] for m in matches]
        except KeyError:
            matched_docs = [Doc(**m.metadata) for m in matches]
        if len(matched_docs) == 0:
            return set()
        # this only works for gpt-4 (in my testing)
        try:
            if (
                rerank is None
                and get_llm_name(cast(BaseLanguageModel, self.llm)).startswith("gpt-4")
                or rerank is True
            ):
                chain = make_chain(
                    self.prompts.select,
                    cast(BaseLanguageModel, self.llm),
                    skip_system=True,
                )
                papers = [f"{d.docname}: {d.citation}" for d in matched_docs]
                result = await chain.arun(  # type: ignore
                    question=query,
                    papers="\n".join(papers),
                    callbacks=get_callbacks("filter"),
                )
                return set([d.dockey for d in matched_docs if d.docname in result])
        except AttributeError:
            pass
        return set([d.dockey for d in matched_docs])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.texts_index is not None and self.index_path is not None:
            state["texts_index"].save_local(self.index_path)
        del state["texts_index"]
        del state["doc_index"]
        return {"__dict__": state, "__fields_set__": self.__fields_set__}

    def __setstate__(self, state):
        object.__setattr__(self, "__dict__", state["__dict__"])
        object.__setattr__(self, "__fields_set__", state["__fields_set__"])
        try:
            self.texts_index = FAISS.load_local(self.index_path, self.embeddings)
        except Exception:
            # they use some special exception type, but I don't want to import it
            self.texts_index = None
        self.doc_index = None

    def _build_texts_index(self, keys: Optional[Set[DocKey]] = None):
        if keys is not None and self.jit_texts_index:
            del self.texts_index
            self.texts_index = None
        if self.texts_index is None:
            texts = self.texts
            if keys is not None:
                texts = [t for t in texts if t.doc.dockey in keys]
            if len(texts) == 0:
                return
            raw_texts = [t.text for t in texts]
            text_embeddings = [t.embeddings for t in texts]
            metadatas = [t.dict(exclude={"embeddings", "text"}) for t in texts]
            self.texts_index = FAISS.from_embeddings(
                # wow adding list to the zip was tricky
                text_embeddings=list(zip(raw_texts, text_embeddings)),
                embedding=self.embeddings,
                metadatas=metadatas,
            )

    def build_doc_index(self):
        from collections import namedtuple

        def docDecoder(dictobj):
            return namedtuple('Doc', dictobj.keys())(*dictobj.values())

        if self.doc_index is None:
            return
        cursor = None
        # Todo: Make it 1000
        batch_size = 100
        while True:
            docs, cursor = self.doc_index.get_objects(
                properties=['dockey'],
                limit = batch_size,
                cursor = cursor)

            for doc in docs:
                new_doc = json.loads(doc.page_content, object_hook=docDecoder)
                self.docs[doc.metadata['dockey']] = new_doc
                self.docnames.add(new_doc.docname)

            if len(docs) < batch_size:
                break

    def clear_memory(self):
        """Clear the memory of the model."""
        if self.memory_model is not None:
            self.memory_model.clear()

    def category_filter_get(self, state_category: Tuple[str], designation_category: Tuple[str], topics: Optional[Tuple[str]] = None):
        category_filter = None


        
        if state_category and designation_category:
            # if the designation is broker add consumer to the designation category
            if "Broker" in designation_category:
                designation_category = set(designation_category)
                designation_category.add("Consumer")

            # add general to the state category irrespective of state.
            state_category = set(state_category)
            state_category.add("General")

            if topics:
                if topics[0] == "All" or "Broker" in designation_category:
                    topics = ("General", "Eligibility", "Enrollments", "Applications", "Account Tasks", "ACA")

                category_filter = {
                    "operator": "And",
                    "operands": [{
                        "path": ["state_category"],
                        "operator": "ContainsAny",
                        "valueText": list(state_category)
                    }, {
                        "path": ["designation_category"],
                        "operator": "ContainsAny",
                        "valueText": list(designation_category)
                    }, {
                        "path": ["topic"],
                        "operator": "ContainsAny",
                        "valueText": list(topics)
                    }]
                }
            else:
                category_filter = {
                    "operator": "And",
                    "operands": [{
                        "path": ["state_category"],
                        "operator": "ContainsAny",
                        "valueText": list(state_category)
                    }, {
                        "path": ["designation_category"],
                        "operator": "ContainsAny",
                        "valueText": list(designation_category)
                    }]
                }


        return category_filter


    def get_evidence(
        self,
        answer: Answer,
        k: int = 10,
        max_sources: int = 5,
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
        detailed_citations: bool = False,
        disable_vector_search: bool = False,
        disable_answer: bool = False,
        reranker: Optional[str] = "None"
    ) -> Answer:
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
                get_callbacks=get_callbacks,
                detailed_citations=detailed_citations,
                disable_vector_search=disable_vector_search,
                disable_answer=disable_answer,
            )
        )

    def filter_unique_matches(self, matches_with_score):
        scores = sorted([m[1] for m in matches_with_score], reverse=True)
        matches_with_score = sorted(matches_with_score, key=lambda tup: tup[1], reverse=True)
        matches = [match_with_score[0] for match_with_score in matches_with_score]
        
        new_matches = []
        new_scores = []
        unique_set = set()

        for m, score in zip(matches, scores):
            # the relevant vectors are already in order, just sorting them
            relevant_vectors = tuple(sorted(m.metadata["relevant_vectors"]))
            if relevant_vectors not in unique_set:
                new_matches.append(m)
                new_scores.append(score)
                unique_set.add(relevant_vectors)

        return new_matches, new_scores


    def get_followon_questions(self,answer, matches, max_sources):
        questions = []
        idx = 0
        while len(set(questions)) < max_sources and idx < len(matches):
            if matches[idx].metadata['follow_on_question']:
                embed_text = matches[idx].metadata['embed_text'][:-5] + "?"
                if answer.question not in embed_text and embed_text not in questions:
                    questions.append(embed_text)

            idx += 1
        
        return questions


    async def aget_evidence(
        self,
        answer: Answer,
        k: int = 10,  # Number of vectors to retrieve
        max_sources: int = 5,  # Number of scored contexts to use
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
        detailed_citations: bool = False,
        disable_vector_search: bool = False,
        disable_answer: bool = False,
        reranker: Optional[str] = "None",
        trace_id: Optional[str] = None,
        state_category: Optional[Tuple[str]] = None,
        designation_category: Optional[Tuple[str]] = None,
        topic: Optional[Tuple[str]] = None,
        follow_on_questions: Optional[bool] = False,
    ) -> Answer:
        if disable_vector_search:
            k = k * 10000
        if len(self.docs) == 0 and self.doc_index is None:
            return answer
        self._build_texts_index(keys=answer.dockey_filter)
        if self.texts_index is None:
            return answer
        self.texts_index = cast(VectorStore, self.texts_index)
        _k = k
        if answer.dockey_filter is not None:
            _k = k * 10  # heuristic
        if marginal_relevance:
            matches = self.texts_index.max_marginal_relevance_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        else:
            # calculate time taken by similarity_search_with_score in milliseconds
            start_time = datetime.now()
            logging.trace(f"state_category:{state_category} designation_category:{designation_category} topics:{topic}")
            category_filter = self.category_filter_get(state_category, designation_category, topic)
            logging.trace(f"weaviate category filter:{category_filter}")
            logging.trace(f"trace_id:{trace_id} category_filter:{category_filter}")

            matches_with_score = self.texts_index.similarity_search_with_score(
                answer.question, k=_k, fetch_k=5 * _k,
                where_filter=category_filter
            )
            logging.trace(f"length of matches with score: {len(matches_with_score)}")
            end_time = datetime.now()
            logging.trace(f"trace_id:{trace_id} vector-search-time:{(end_time - start_time).microseconds / 1000} ms")

            if not matches_with_score:
                raise NoMatchesFoundException("No matches found for the given query")

            # matches_with_score is a list of tuples (doc, score)
            # fetch all the scores in a list, sort them in descending order

            matches, scores = self.filter_unique_matches(matches_with_score)

            rank = 1
            num_of_log_entries = 10
            for m, score in zip(matches[:num_of_log_entries], scores[:num_of_log_entries]):
                vector_id = m.metadata["_additional"]["id"]
                logging.trace(f"trace_id:{trace_id} rank:{rank} id:{vector_id}, score:{score:.2f}"
                              f" doc:{json.loads(m.metadata['doc'])['docname']}"
                              f" doc source: {m.metadata['doc_source']}-{m.metadata['state_category']}")
                rank += 1

        for m in matches:
            if isinstance(m.metadata["doc"], str):
                m.metadata["doc"] = json.loads(m.metadata["doc"])

        questions = []
        if follow_on_questions:
            questions = self.get_followon_questions(answer, matches, max_sources)

        answer.follow_on_questions = questions

        # ok now filter
        #if answer.dockey_filter is not None:
        #    matches = [
        #        m
        #        for m in matches
        #        if m.metadata["doc"]["dockey"] in answer.dockey_filter
        #    ]

        # check if it is deleted
        matches = [
            m
            for m in matches
            if m.metadata["doc"]["dockey"] not in self.deleted_dockeys
        ]

        # check if it is already in answer
        cur_names = [c.text.name for c in answer.contexts]
        matches = [m for m in matches if m.metadata["name"] not in cur_names]

        # now fnally cut down
        matches = matches[:max_sources]
        
        # create score for each match
        for i, match in enumerate(matches):
            match.metadata["score"] = 0

        # def get_next_context(source):
        #     doc_vector_ids = source.metadata['doc_vector_ids']
        #     parent_chunk = ''
        #     vid = ''
        #     if len(doc_vector_ids) > 3:
        #         sid = source.metadata['_additional']['id']
        #         sid_index = doc_vector_ids.index(sid)
        #
        #         if not sid_index:
        #             vid = doc_vector_ids[sid_index + 3]
        #         elif sid_index > 0 and sid_index < (len(doc_vector_ids) - 3):
        #             vid = doc_vector_ids[sid_index + 2]
        #
        #         if vid != '':
        #             data_object = self.texts_index._client.data_object.get_by_id(
        #                 vid,
        #                 class_name=self.texts_index._index_name,
        #             )
        #
        #             parent_chunk = data_object['properties']['parent_chunk']
        #
        #     return parent_chunk
        #
        # next_contexts = [get_next_context(m) for m in matches]

        async def process(match):
            callbacks = get_callbacks("evidence:" + match.metadata["name"])
            summary_chain = make_chain(
                self.prompts.summary,
                self.summary_llm,
                memory=self.memory_model,
                system_prompt=self.prompts.system[answer.system],
            )
            # This is dangerous because it
            # could mask errors that are important- like auth errors
            # I also cannot know what the exception
            # type is because any model could be used
            # my best idea is see if there is a 4XX
            # http code in the exception
            try:
                citation = match.metadata["doc"]["citation"]
                if detailed_citations:
                    citation = match.metadata["name"] + ": " + citation
                if self.prompts.skip_summary:
                    context = match.page_content
                else:
                    dockey = match.metadata["doc"]["dockey"]
                    logging.debug(f"dockey: {dockey}, input chunk: \n{match.page_content}\n")
                    context = await summary_chain.arun(
                        question=answer.question,
                        # Add name so chunk is stated
                        citation=citation,
                        summary_length=answer.summary_length,
                        text=match.page_content,
                        callbacks=callbacks,
                    )
                    logging.debug(f"dockey: {dockey}, output context:\n {context}\n")
            except Exception as e:
                if guess_is_4xx(str(e)):
                    return None
                raise e
            if "not applicable" in context.lower() or "not relevant" in context.lower():
                return None

            c = Context(
                context=context,
                text=Text(
                    text=match.page_content,
                    name=match.metadata["name"],
                    doc=Doc(**match.metadata["doc"]),
                ),
                vector_id=match.metadata["_additional"]["id"],
                score=get_score(context),
            )
            return c

        if disable_answer:
            contexts = [
                Context(
                    context=match.page_content,  # + next_contexts[idx],
                    score=10,
                    weaviate_score=scores[idx],
                    text=Text(
                        text=match.page_content,  # + next_contexts[idx],
                        name=match.metadata["name"],
                        doc=Doc(**match.metadata["doc"]),
                        vector_id=match.metadata["_additional"]["id"],
                        ext_path=match.metadata["ext_path"],
                        dockey=match.metadata.get("dockey"),
                        doc_source=match.metadata["doc_source"][0],
                    ),
                    vector_id=match.metadata["_additional"]["id"]
                )
                for idx, match in enumerate(matches)
            ]

        else:
            if reranker:
                start_time = datetime.now()
                query_and_matches = [[answer.question, m.page_content] for m in matches]
                model = CrossEncoder(
                    # model_name="BAAI/bge-reranker-large", max_length=512
                    model_name="BAAI/bge-reranker-v2-m3", max_length=1024
                )
                scores = model.predict(query_and_matches)
                for match, score in zip(matches, scores):
                    match.metadata['score'] = score

                matches = sorted(matches, key=lambda x: -x.metadata['score'] if x.metadata['score'] else 0)

                for m in matches:
                    vector_id = m.metadata["_additional"]["id"]
                    logging.trace(f"trace_id:{trace_id} rerank-vectorid:{vector_id} reranker score:{m.metadata['score']}")

                end_time = datetime.now()
                logging.trace(f"trace_id:{trace_id} reranker-time:{(end_time - start_time).microseconds / 1000}ms")

                for i, match in enumerate(matches):
                    logging.info(f"content: {match.page_content[:32]} {match.metadata['score']}")

                contexts = [
                    Context(
                        context=match.page_content,
                        text=Text(
                            text=match.page_content,
                            name=match.metadata["name"],
                            doc=Doc(**match.metadata["doc"]),
                        ),
                        vector_id=match.metadata["_additional"]["id"],
                        score=match.metadata['score'],
                    )
                    for match in matches
                ]

            else:
                results = await gather_with_concurrency(
                    self.max_concurrent, *[process(m) for m in matches]
                )
                # filter out failures
                contexts = [c for c in results if c is not None]

    
        answer.contexts = sorted(
            contexts + answer.contexts, key=lambda x: x.score, reverse=True
        )
        answer.contexts = answer.contexts[:max_sources]
        context_str = "\n\n".join(
            [
                f"{c.text.name}: {c.context}"
                + (f"\n\n Based on {c.text.doc.citation}" if detailed_citations else "")
                for c in answer.contexts
            ]
        )

        valid_names = [c.text.name for c in answer.contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        answer.context = context_str
        return answer

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt="about 200 words",
        marginal_relevance: bool = True,
        answer: Optional[Answer] = None,
        key_filter: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
        disable_answer: bool = False,
        reranker: Optional[str] = "None",
        trace_id: Optional[str] = None,
    ) -> Answer:
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aquery(
                query,
                k=k,
                max_sources=max_sources,
                length_prompt=length_prompt,
                marginal_relevance=marginal_relevance,
                answer=answer,
                key_filter=key_filter,
                get_callbacks=get_callbacks,
                disable_answer=disable_answer,
                reranker=reranker,
                trace_id=trace_id,
            )
        )

    def get_reference_dict(self, references):
        dict_ = {"references": []}
        i = 1
        ref = ''
        url = ''
        while(re.findall(rf'{i}. .*', references)) != []:
            d = {}
            ref_str = re.findall(rf'{i}. .*', references)[0][3:]
            if ref_str.startswith('['):
                ref = re.findall(r'\[.*\]', ref_str)[0][1:][:-1]
                url = re.findall(r'\(.*\)', ref_str)[0][1:][:-1]
            else:
                ref = ref_str
                url = ref_str
            d["rank"] = i
            d["ref"] = ref
            d["url"] = url
            dict_["references"].append(d)
            i = i + 1

        return dict_


    async def faq_aget_evidence(self, answer, k, trace_id, state_category, designation_category, topic, follow_on_questions, max_sources, stream_json):
        category_filter = self.category_filter_get(state_category, designation_category)
        logging.trace(f"trace_id:{trace_id} category_filter:{category_filter}")
       
        matches_with_score = []
        try:
            matches_with_score = self.cache_index.similarity_search_with_score(
                answer.question, k=k, fetch_k=k,
                where_filter=category_filter
            )
        except Exception as e:
            print(f"ERROR: error in searching in cache, {e}")
            answer.faq_vectorstore_score = 0.0
            # return answer
       
        if not matches_with_score:
            answer.faq_vectorstore_score = 0.0

        if matches_with_score:
            answer.faq_feedback = matches_with_score[0][0].metadata['feedback']
            answer.faq_vectorstore_score = matches_with_score[0][1]
            answer.validated = matches_with_score[0][0].metadata['validated']
            
            if (answer.faq_feedback in ['positive', 'negative'] and answer.faq_vectorstore_score >= 0.90) or (answer.faq_vectorstore_score >= 0.98):
                if answer.faq_feedback == 'negative':
                    answer.answer = matches_with_score[0][0].metadata['feedback_answer']
                    answer.references = matches_with_score[0][0].metadata['feedback_sources']
                elif answer.faq_feedback == 'positive':
                    answer.answer = matches_with_score[0][0].page_content if matches_with_score[0][0].metadata['feedback_answer'] == '' else matches_with_score[0][0].metadata['feedback_answer']
                    answer.references = matches_with_score[0][0].metadata['references']
                else:
                    answer.answer = matches_with_score[0][0].page_content
                    answer.references = matches_with_score[0][0].metadata['references']

                if stream_json:
                    answer.ref_str = answer.references
                    answer.references = self.get_reference_dict(answer.references)
                    answer.references["id"] = trace_id

                answer.faq_vector_id = matches_with_score[0][0].metadata['_additional']['id']
                answer.parent_req_id = matches_with_score[0][0].metadata['trace_id']
                answer.faq_doc = matches_with_score[0][0].metadata['doc']
                answer.trace_id = trace_id
                answer.faq_match_question = matches_with_score[0][0].metadata['question']

                questions = []
                if follow_on_questions:
                    category_filter = self.category_filter_get(state_category, designation_category, topic)
                    _k = 10
                    matches_with_score = self.texts_index.similarity_search_with_score(
                        answer.question, k=_k, fetch_k=5 * _k,
                        where_filter=category_filter
                    )
                    matches, scores = self.filter_unique_matches(matches_with_score)

                    for m in matches:
                        if isinstance(m.metadata["doc"], str):
                            m.metadata["doc"] = json.loads(m.metadata["doc"])
                    
                    questions = self.get_followon_questions(answer, matches, max_sources)

                answer.follow_on_questions = questions
       
        return answer


    async def vectorstore_call(
        self,
        query: str,
        k: Optional[int] = 10,
        max_sources: Optional[int] = 5,
        marginal_relevance: Optional[bool] = True,
        answer: Optional[Answer] = None,
        length_prompt: Optional[str] = "about 100 words",
        key_filter: Optional[bool] = None,
        get_callbacks: Optional[CallbackFactory] = lambda x: None,
        disable_answer: Optional[bool] = False,
        reranker: Optional[str] = "None", # Replace this with enum
        trace_id: Optional[str] = None,
        state_category: Optional[Tuple[str]] = None,
        designation_category: Optional[Tuple[str]] = None,
        topic: Optional[Tuple[str]] = None,
        enable_cache: Optional[bool] = False,
        anchor_flag: Optional[bool] = False,
        follow_on_questions = False,
        stream_json: Optional[bool] = False,
    ) -> Answer:
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
            answer.trace_id = trace_id

        if len(answer.contexts) == 0:
            # this is heuristic - k and len(docs) are not
            # comparable - one is chunks and one is docs
            if key_filter or (key_filter is None and len(self.docs) > k):
                keys = await self.adoc_match(
                    answer.question, get_callbacks=get_callbacks
                )
                if len(keys) > 0:
                    answer.dockey_filter = keys

            if enable_cache:
                answer = await self.faq_aget_evidence(
                    answer,
                    k=k,
                    trace_id=trace_id,
                    state_category=state_category,
                    designation_category=designation_category,
                    topic=topic,
                    follow_on_questions=follow_on_questions,
                    max_sources=max_sources,
                    stream_json=stream_json,
                )
            else:
                answer = await self.aget_evidence(
                    answer,
                    k=k,
                    max_sources=max_sources,
                    marginal_relevance=marginal_relevance,
                    get_callbacks=get_callbacks,
                    disable_answer=disable_answer,
                    reranker=reranker,
                    trace_id=trace_id,
                    state_category=state_category,
                    designation_category=designation_category,
                    topic=topic,
                    follow_on_questions=follow_on_questions,
                )

        return answer


    async def rewrite_query(
        self,
        query: str,
        get_callbacks: CallbackFactory = lambda x: None,
        trace_id: Optional[str] = None,
        state_category: Optional[Tuple[str]] = None,
        designation_category: Optional[Tuple[str]] = None,
        topic: Optional[Tuple[str]] = None,
    ) ->  dict:  

        CONFIDENCE_THRESHOLD = 5 #out of 10 for a rewrite
        def remove_suffix(text, match):
            index = text.rfind(match)
            if index == -1:
                return text
            else:
                return text[:index].strip()

        def extract_followup_questions(text):
            """
            Extracts the 'Followup questions' section from the given text and returns it as a JSON string.
            
            Parameters:
                text (str): The input text containing potential 'Followup questions'.
                
            Returns:
                str or None: JSON string of follow-up questions if found and parsed successfully; otherwise, None.
            """
            # Define a regex pattern to locate 'Followup questions:' followed by a JSON array
            pattern = r'(Follow[-\s]?up questions):\s*(\[[^\]]*\])'
            
            # Search for the pattern using regex
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if not match:
                # 'Followup questions:' section not found
                return None
            
            # Extract the JSON array string from the matched group
            followup_text = match.group(2)
            
            try:
                # Parse the extracted string to verify it's valid JSON
                followup_questions = eval(followup_text)
                
                # Convert the Python object back to a formatted JSON string
                #followup_json = json.dumps(followup_questions, indent=4)
                
                return followup_questions
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                print("Error decoding JSON:", e)
                return None


        def extract_rewritten_questions(derived_ctx):
            import re
            # Use regex to find the first section enclosed in square brackets
            pattern = re.compile(r'\[.*?\]', re.DOTALL)
            match = pattern.search(derived_ctx)

            if match:
                derived_ctx = match.group(0)
                return eval(derived_ctx)
            else:
                # rewrite formal failures: Use the original question. No rewrite
                return None


        def add_followup_questions(answer, questions, skip1=False):
            for q in questions:
                if skip1:
                    skip1 = False
                elif q['confidence_score'] >= CONFIDENCE_THRESHOLD and \
                    answer.question != q['question']:
                        answer.follow_on_questions.append(f"{q['question']}/norewrite")

        followup_questions = None
        answer = Answer(question=query.strip())
        answer.trace_id = trace_id
        answer.finline_response = False
        answer.follow_on_questions = []
        answer.state_category = state_category[0] if state_category else 'General'

        if answer.question.endswith(("/norewrite", "/norewrite?", "/norewrite ?")):
            # Todo: Use LLM to just create topic & category
            answer.question = remove_suffix(answer.question, "/norewrite")
            return answer

        rewrite_chain = make_chain(
            self.prompts.rewrite[answer.state_category],
            cast(BaseLanguageModel, self.llm),
            memory=self.memory_model,
            system_prompt=self.prompts.system[answer.state_category],
        )
        start_time = datetime.now()
        derived_ctx = ""

        try:
            derived_ctx = await rewrite_chain.arun(
                scenario=query,
                json_format='[{"question": "...","group": "...","topic": "...", "confidence_score": "..."}, {"question": "...","group": "...","topic": "...", "confidence_score": "..."}]'
                #callbacks=get_callbacks("rewrite"),
            )
        except Exception as e:
            answer_text = str(e)
            logging.trace(f"trace_id:{trace_id}, rewrite_chain failure: {answer_text}")
            # rewrite format failures: Use the original question
            return answer

        end_time = datetime.now()

        logging.trace(f"trace_id:{trace_id} rewrite-time:{(end_time - start_time).microseconds / 1000}ms")
        logging.trace(f"trace_id:{trace_id} derived_json: {derived_ctx}")
        try: 
            if derived_ctx != "":
                derived = extract_rewritten_questions(derived_ctx)
                # Extract and convert follow-up questions to JSON
                followup_questions = extract_followup_questions(derived_ctx)
                nquestions = len(derived)

                if nquestions == 0 or (nquestions and derived[0]['confidence_score'] < CONFIDENCE_THRESHOLD):
                    # can't answer
                    answer.finline_response = True
                    answer.answer =f"\nThe original question is ambiguous. "\
                                    "Please rephrase or escalate to supervisor. \n\n"
                    
                else:
        
                    for idx, q in enumerate(derived):
                        if q['question'] == answer.question:
                            answer.metadata = {'category':q['group'], 'topic':q['topic']}
                            del derived[idx]
                            break

                    if answer.metadata is None:
                        answer.question = derived[0]['question']
                        answer.metadata = {'category':derived[0]['group'], 'topic':derived[0]['topic']}
                        add_followup_questions(answer, derived, skip1=True)

                    elif len(derived):
                        add_followup_questions(answer, derived)
                
                if followup_questions:
                    add_followup_questions(answer, followup_questions)

                return answer
            
        except Exception as e:
            answer_text = str(e)
            logging.trace(f"trace_id:{trace_id}, rewrite json failure: {answer_text}")
            answer.answer = "I can't answer this question. Please rephrase the question or escalate to supervisor."

        return answer


    async def aquery(
        self,
        answer: Answer,
        get_callbacks: CallbackFactory = lambda x: None,
        trace_id: Optional[str] = None,
        stream_json: Optional[bool] = False,
        securellm: Optional[bool] = False,
    ) ->  Answer:
        if self.prompts.pre is not None:
            chain = make_chain(
                self.prompts.pre,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system[answer.state_category],
            )
            pre = await chain.arun(
                question=answer.question, callbacks=get_callbacks("pre")
            )
            answer.context = pre + "\n\n" + answer.context


        bib = dict()
        bib_str = [] if stream_json else ""
        ref_dict = []
        ref_str = '\n\n**References:**\n\n'
        for i, c in enumerate(answer.contexts):
            name = c.text.name
            citation = c.text.doc.citation

            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            #if name_in_text(name, answer_text):
            #   bib[name] = citationi
            SHARE_POINT_URL = "https://giprod.sharepoint.com/:b:/r/sites/TrainingTeam/Shared%20Documents/"

            if c.text.ext_path:
                if c.text.doc_source.lower() == 'external' or c.text.ext_path.startswith('http'):
                    url = c.text.ext_path
                else:
                    url = SHARE_POINT_URL + quote(c.text.ext_path)

                if stream_json:
                    bib_str.append({"rank": i+1,"ref":f"{name}", "url":f"{url}"})
                else:
                    bib_str += f"\n {i+1}. [{name}]({url})"
                ref_str += f"\n {i+1}. [{name}]({url})"
                ref_dict.append({"rank": i+1,"ref":f"{name}", "url":f"{url}", "doc_source": c.text.doc_source.lower()})
            else:
                if name != citation:
                    if stream_json:
                        bib_str.append({"rank":i+1, "ref":f"{name}", "citation": f"{citation}"})
                    else:
                        bib_str += f"\n {i+1}. {name}: {citation}"
                    ref_str += f"\n {i+1}. [{name}]({citation})"
                    ref_dict.append({"rank":i+1, "ref":f"{name}", "citation": f"{citation}", "doc_source": c.text.doc_source.lower()})
                else:
                    if stream_json:
                        bib_str.append({"rank":i+1, "ref":f"{citation}"})
                    else:
                        bib_str += f"\n {i+1}. {citation}"
                    ref_str += f"\n {i+1}. [{citation}]()"
                    ref_dict.append({"rank":i+1, "ref":f"{citation}", "doc_source": c.text.doc_source.lower()})
        
        answer.ref_str = ref_str
        if securellm:
            tags = json.loads(self.llm.model_kwargs['headers']['x-sgpt-tags'])
            tags['debug_properties']['references'] = ref_str
            tags['debug_properties']['ref_dict'] = ref_dict 
            tags = json.dumps(tags)
            self.llm.model_kwargs['headers']['x-sgpt-tags'] = tags

        if len(answer.context) < 10 and not self.memory:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            start_time = datetime.now()
            callbacks = get_callbacks("answer")
            if self.memory_model:
                memory_str = str(self.memory_model.load_memory_variables({})["memory"])
                logging.trace(f"trace_id:{trace_id} conversation_history:{memory_str}")

            if self.memory_model and self.memory_model.buffer:
                followup_chain = make_chain(
                    self.prompts.followup,
                    cast(BaseLanguageModel, self.llm),
                    # memory=self.memory_model,
                    system_prompt=self.prompts.system[answer.state_category]
                )
                previous_question = self.memory_model.buffer[-2].content
                try:
                    followup_question = await followup_chain.arun(
                        previous_question=previous_question,
                        question=answer.question,
                        # callbacks=callbacks,
                    )
                except Exception as e:
                    followup_question = str(e)
                answer.question = followup_question
                logging.trace(f"trace_id:{trace_id} follow-up:{answer.question}")

            qa_chain = make_chain(
                self.prompts.qa,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system[answer.state_category]
            )

            try:
                # logging.trace(f"trace_id:{trace_id} context:{answer.context}")
                answer_text = await qa_chain.arun(
                    context=answer.context,
                    answer_length=answer.answer_length,
                    question=answer.question,
                    callbacks=callbacks,
                )
            except Exception as e:
                answer_text = str(e)

            end_time = datetime.now()
            logging.trace(f"trace_id:{trace_id} qa-time:{(end_time - start_time).microseconds / 1000}ms")
        # it still happens
        if "(Example2012)" in answer_text:
            answer_text = answer_text.replace("(Example2012)", "")

        # bib_str = [] if stream_json else ""
        # for i, c in enumerate(answer.contexts):
        #     name = c.text.name
        #     citation = c.text.doc.citation

        #     # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
        #     #if name_in_text(name, answer_text):
        #     #   bib[name] = citationi
        #     SHARE_POINT_URL = "https://giprod.sharepoint.com/:b:/r/sites/TrainingTeam/Shared%20Documents/"

        #     if c.text.ext_path:
        #         if c.text.doc_source.lower() == 'external' or c.text.ext_path.startswith('http'):
        #             url = c.text.ext_path
        #         else:
        #             url = SHARE_POINT_URL + quote(c.text.ext_path)

        #         if stream_json:
        #             bib_str.append({"rank": i+1,"ref":f"{name}", "url":f"{url}"})
        #         else:
        #             bib_str += f"\n {i+1}. [{name}]({url})"
        #     else:
        #         if name != citation:
        #             if stream_json:
        #                 bib_str.append({"rank":i+1, "ref":f"{name}", "citation": f"{citation}"})
        #             else:
        #                 bib_str += f"\n {i+1}. {name}: {citation}"
        #         else:
        #             if stream_json:
        #                 bib_str.append({"rank":i+1, "ref":f"{citation}"})
        #             else:
        #                 bib_str += f"\n {i+1}. {citation}"

        formatted_answer = f"Question: {answer.question}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = {"references":bib_str, "id":trace_id} if stream_json else bib_str
        answer.ref_dict = ref_dict

        if self.prompts.post is not None:
            chain = make_chain(
                self.prompts.post,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system[answer.state_category]
            )
            post = await chain.arun(**answer.dict(), callbacks=get_callbacks("post"))
            answer.answer = post
            answer.formatted_answer = f"Question: {answer.question}\n\n{post}\n"
            if len(bib) > 0:
                answer.formatted_answer += f"\nReferences\n\n{bib_str}\n"

        if self.memory_model is not None and not anchor_flag:
            answer.memory = self.memory_model.load_memory_variables(inputs={})["memory"]
            self.memory_model.save_context(
                {"Question": answer.question}, {"Answer": answer.answer}
            )
            self.memory_model.clear(self.memory_model.k)

        return answer
