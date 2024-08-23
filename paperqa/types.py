from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, validator

from .prompts import (
    citation_prompt,
    default_system_prompt,
    system_prompts,
    qa_prompt,
    select_paper_prompt,
    summary_prompt,
    rewrite_prompts,
    followup_system_prompt
)

StrPath = Union[str, Path]
DocKey = Any
CBManager = Union[AsyncCallbackManagerForChainRun, CallbackManagerForChainRun]
CallbackFactory = Callable[[str], Union[None, List[BaseCallbackHandler]]]


class Doc(BaseModel):
    docname: str
    citation: str
    dockey: DocKey


class Text(BaseModel):
    text: str
    text_length: int = 0
    name: str
    doc: Doc
    embeddings: Optional[List[float]] = None
    dockey: Any
    token_count: Optional[int] = 0
    parent_chunk: Optional["Text"] = None
    vector_id: Optional[str] = None
    reranker_vector_id: Optional[str] = None
    base_vector_id: Optional[str] = None
    embed_text: Optional[str] = None
    relevant_vectors: Optional[List[str]] = []
    csv_text: Optional[str] = None
    doc_vector_ids: Optional[List[str]] = None
    page_text: Optional[str] = None
    page_no: Optional[int] = None
    is_table: Optional[bool] = False
    is_pdf: Optional[bool] = False
    state_category: Optional[List[str]] = None
    designation_category: Optional[List[str]] = None
    topic: Optional[List[str]] = None
    ext_path: Optional[str] = None
    doc_source: Optional[str] = None
    follow_on_question: Optional[bool] = None

class Faq_Text(BaseModel):
    question: str
    answer: str
    trace_id: str
    doc: Doc
    vector_id: str
    references: str 
    state_category: List[str]
    designation_category: List[str]
    embeddings: Optional[List[float]] = None
    date: Optional[str] = None
    feedback: Optional[str] = None
    feedback_answer: Optional[str] = None
    feedback_sources: Optional[str] = None

class PromptCollection(BaseModel):
    summary: PromptTemplate = summary_prompt
    qa: PromptTemplate = qa_prompt
    followup: Optional[PromptTemplate] = followup_system_prompt
    select: PromptTemplate = select_paper_prompt
    cite: PromptTemplate = citation_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None
    rewrite: dict = rewrite_prompts
    system: dict = system_prompts
    skip_summary: bool = False

    @validator("summary")
    def check_summary(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(summary_prompt.input_variables)):
            raise ValueError(
                f"Summary prompt can only have variables: {summary_prompt.input_variables}"
            )
        return v

    @validator("qa")
    def check_qa(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(qa_prompt.input_variables)):
            raise ValueError(
                f"QA prompt can only have variables: {qa_prompt.input_variables}"
            )
        return v

    @validator("select")
    def check_select(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(
                set(select_paper_prompt.input_variables)
        ):
            raise ValueError(
                f"Select prompt can only have variables: {select_paper_prompt.input_variables}"
            )
        return v

    @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(v.input_variables).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v

    @validator("followup")
    def check_followup(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(followup_system_prompt.input_variables)):
            raise ValueError(
                f"followup_system_prompt prompt can only have variables: {summary_prompt.input_variables}"
            )
        return v


class Context(BaseModel):
    """A class to hold the context of a question."""

    context: str
    text: Text
    vector_id: str = ''
    score: int = 5
    weaviate_score: float = 0.0


def __str__(self) -> str:
    """Return the context as a string."""
    return self.context


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    system: str = "General"
    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Context] = []
    references: str = ""
    ref_dict: dict = {}
    ref_str: str = '' 
    formatted_answer: str = ""
    dockey_filter: Optional[Set[DocKey]] = None
    summary_length: str = "about 200 words"
    answer_length: str = "about 300 words"
    memory: Optional[str] = None
    # these two below are for convenience
    # and are not set. But you can set them
    # if you want to use them.
    cost: Optional[float] = None
    token_counts: Optional[Dict[str, List[int]]] = None
    trace_id: Optional[str] = None
    faq_vectorstore_score: Optional[float] = None
    faq_vector_id: Optional[str] = ''
    parent_req_id: Optional[str] = ''
    faq_doc: Optional[Doc] = None
    faq_match_question: Optional[str]=None
    faq_feedback: Optional[str]=None
    follow_on_questions: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None
    finline_response: bool = False


    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer
