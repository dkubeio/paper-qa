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
    input_variables=["context", "answer_length", "question", "json_format"],
    template="Write an answer in {answer_length} "
    "for the question below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "I cannot answer". '
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). \n"
    "Include confidence score of the generated answer on the scale of 1 to 10 \n"
    "Do not explain confidence score. \n"
    # "Provide the output in given response_format. \n"
    # "for examples. \n"
    # "Example 1 \n"
    # "Question: what are the customer requirements for a term 03?\n(Give answer for question asked above in following JSON format {\"answer\": ['type:str'], \"confidence_score\": ['type:int'], \"sources\": ['type:str']})"
    # "Output: {\"answer\": [\"A customer, who has been enrolled in coverage through a state-based exchange (e.g., Pennie), becomes enrolled in Medicaid or the Children\'s Health Insurance Program (CHIP) with a retroactive coverage start date. The customer provides documentation showing which family members are enrolled in Medicaid or CHIP, the date the Medicaid or CHIP coverage began, and the date the notice was generated and/or mailed to the customer. The customer reports the issue within 60 days of receipt of their eligibility notice from Medicaid or CHIP. The date of the notice of the customer\'s eligibility for Medicaid or CHIP must be no more than 60 days prior to the date the customer first reported the issue to the exchange.\"],\"confidence_score\": [9],\"sources\": [\"Term-03.pdf pages 1\"]}" 
    # "Example 2 \n"
    # "Question: if the customer requests their information to be sent to MA for review, how do we do that?\n(Give answer for question asked above in following JSON format {\"answer\": ['type:str'], \"confidence_score\": ['type:int'], \"sources\": ['type:str']})"
    # "Output: [{\"answer\": [\"Create a ticket for the Issuer team with the subject \'Customer Information Review Request: (Carrier Name)\', Ticket Type: \'Issuer\', Ticket Subtype: \'Customer Information Review\', Created For: (Customer Name), Priority: \'Medium\', Ticket Description: \'Customer has requested their information to be sent to MA for review.\'], \"confidence_score\": [9], \"sources\": [\"Customer Account Creation - CSR View.pdf pages 5\", \"Ticket Templates - PA.pdf pages 1\"]}\n"
    # "Your response should only be in JSON. No text responses. \n"
    # "Strictly follow the following JSON format for the answer. \n"
    # "For eg. \n"
    # "Question: What is ACA?"
    # "Context (with relevance scores): \"Hello, today we are going to learn about ACA. ACA is Affordable Care Act, also called Obamacare.\" \n"
    # "output: {{\"confidence_score\":\"9\", \"Answer\":\"ACA is Affordable care act, also called obamacare.\", \"Explaination\":\"The context provides information about ACA.\"}} \n"
    
    # "Do not add any qualifiers or any explaination of confidentiality score or answer. \n"
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}{json_format}\n"
    "Output: \n"
)

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


default_json_system_prompt = (
    "You are a helpful assistant designed to output JSON. Your response should only be in JSON. No text responses"
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
