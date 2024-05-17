from langchain.prompts import PromptTemplate

'''
summary_prompt = PromptTemplate(
    input_variables=["text", "citation", "question", "summary_length"],
    template="Summarize the text below to help answer a question. "
    "Do not directly answer the question, instead summarize "
    "to give evidence to help answer the question. "
    "Focus on specific details, including numbers, equations, or specific quotes. "
    'Reply "Not applicable" if text is irrelevant. '
    "Use {summary_length}. At the end of your response, provide a score from 1-10 on a newline "
    "indicating relevance to question. Do not explain your score. "
    "\n\n"
    "{text}\n\n"
    "Excerpt from {citation}\n"
    "Question: {question}\n"
    "Relevant Information Summary:",
)

qa_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template="Write an answer ({answer_length}) "
    "for the question below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "I cannot answer". '
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). \n"
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
)
'''
summary_prompt = PromptTemplate(
    input_variables=["text", "citation", "question", "summary_length"],
    template="Provide a summary of the Text below with sufficient details to help answer the Question below. "
    "Also provide a score from 0-10 with an interval of 1, indicating relevance of the Text below to the Question below. "
    "Do not explain the score. "
    "Provide specific details, including numbers, equations, or specific quotes. "
    'If text is not relevant, Reply "None" in answer with a score of 0. '
    "Limit the answer to {summary_length}. "
    "Output format:\nAnswer:\n- ...\n- ...\nScore:\n- ...\n"
    "\n\n"
    "Text:{text}\n\n"
    "Excerpt from {citation}\n"
    "Question: {question}\n"
    "Output:\n\n",
)

qa_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template="Write an answer in {answer_length} "
    "for the question below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "I cannot answer". '
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). \n"
    "Include confidence score of the generated answer on the scale of 1 to 10 \n"
    "Do not explain Confidence score. \n"
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
)

# qa_prompt = PromptTemplate(
#     input_variables=["context", "answer_length", "question", "json_format"],
#     template="You are an expert Call Center Assistant for Health Insurance market. \n"
#     "Please act as an impartial judge and evaluate quality of Context provided for the User's Question displayed below. \n"
#     "Your evaluation should consider factors such as the accuracy, depth, level of detail, relevance and helpfulness of the Context to answer the User's Question precisely. \n"
#     "Each factor is worth 1 point. \n"
#     "Be objective as possible. \n"
#     # "After providing your Explanation, please rate the context on a scale of the 1 to 5 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[3]]\". \n"
#     "Please rate the context on a scale of the 1 to 5 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[3]]\". \n"

#     "After rating the context now, Write an answer in {answer_length} "
#     "for the user's question below based on the provided context. "
#     # "for the user's question based on the provided context. "
#     "If the context provides insufficient information and the question cannot be directly answered, "
#     'reply "I cannot answer". '
#     "For each part of your answer, indicate which sources most support it "
#     "via valid citation markers at the end of sentences, like (Example2012). \n"
#     # "Context :\n {context}\n"
#     # "User's Question: {question}{json_format}\n"
#     # "Answer: \n\n",
#     "[User's Question] \n"
#     "{question} \n"

#     "[The start of the Context] \n"
#     "{context} \n"
#     "[The end of the Context] \n"

#     "[The start of the Answer] \n"
#     "Answer: \n"
#     "[The end of the Answer] \n"
#     # "[Start of your Explanation] \n"
#     # "Explanation: \n"
#     # "[End of your Explanation] \n"

#     "[Start of Your Rating] \n"
#     "Rating: [[rating]] \n"
#     "[End of your Rating] \n"

#     # "Write an answer in {answer_length} "
#     # "Now, Write an answer in {answer_length} "
#     # "for the user's question below based on the provided context. "
#     # "for the user's question based on the provided context. "
#     # "If the context provides insufficient information and the question cannot be directly answered, "
#     # 'reply "I cannot answer". '
#     # "For each part of your answer, indicate which sources most support it "
#     # "via valid citation markers at the end of sentences, like (Example2012). \n"
#     # "Context :\n {context}\n"
#     # "User's Question: {question}{json_format}\n"
#     # "Answer: \n\n",
#     )

select_paper_prompt = PromptTemplate(
    input_variables=["question", "papers"],
    template="Select papers that may help answer the question below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. '
    "Choose papers that are relevant, from reputable sources, and timely "
    "(if the question requires timely information). \n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:",
)

# We are unable to serialize with partial variables
# so TODO: update year next year
citation_prompt = PromptTemplate(
    input_variables=["text"],
    template="Provide the citation for the following text in MLA Format. The year is 2023\n"
    "{text}\n\n"
    "Citation:",
)

'''
default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them. "
)
'''
default_system_prompt = (
    "You are a Retrieval Augmented Generation chatbot. "
    "Think step by step and answer in a direct and concise tone. "
)

followup_system_prompt = PromptTemplate(
    input_variables=["question", "previous_question"],
    template="You are an expert synthesizer for conversational chat. "
    "The question below is a followup question based on the previous chat. "
    "Please rephrase the question by synthesizing the question and the previous chat. "
    "Make the new question within 25 words."
    "Don't use sources and references for the new question."
    "Don't write anything except the question."
    "Question:  {question}\n\n"
    "Chat: Question: {previous_question}",
)
