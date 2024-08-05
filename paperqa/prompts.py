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

qa_prompt_old = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template="Write an answer in {answer_length} "
    "for the question below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "I cannot answer". '
    "You don't need to directly answer the question. If it is a policy related question, explain the policy "
    "If the question is asking for a procedure, answer the process "
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). \n"
    "Include confidence score of the generated answer on the scale of 1 to 10 \n"
    "Do not explain Confidence score. \n"
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
)

qa_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template="You are an expert Call Center Agent Assist in the public healthcare insurance marketplace. "
    "Your job is to extract relevant context for the user's question. "
    "Never directly answer yes or no, but only provide policy or procedural information from relevant sections "
    "If the context doesn't provide answer, but provides policy for a part of the question, state the policy. "
    "Do not assume anything. Use the context and not any prior learnings. "
    "Please limit the output to 100 words. "
    "Please do not include any explanatory logic or notes. "
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). "
    "Include sources used at the end of the response"
    "Include confidence score of the generated summary on the scale of 1 to 10 \n"
    "Do not explain Confidence score. \n"
    "If the context provides sufficient information reply strictly in the format; Answer: ...\n Sources: ...\n Confidence score: ... "
    "If the context provides insufficient information reply `I cannot answer, Please escalte to supervisor or rephrase the question` and don't provide any logic for deriving this conclusion. "
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}\n"
)
    #"Answer: "
    #"Sources: ",
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
#"Respond in a JSON format as specified below {json_format}. Strictly follow this format and do not include any other text: "
#   "where question is derived from the scenario, group and topic are derived by classifying the derived question. "
rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    template="You are an expert Call Center Agent Assist in the public healthcare insurance marketplace. "
    "Your job is to analyze a customer scenario, derive all the policy or procedure subquestions and classify the questions based on some criteria below. "
    "list each question separately as specified in the json format {json_format} in a single json list. "
    "Each element in the json list should be a json object with 3 key/value pairs, where the keys are question, group and topic. "
    "Strictly follow this format and do not include any other text, explanation or notes "
    "If the scenario is ambiguous and doesn't describe any question, please return n/a for question, otherwise proceed with the classification of question. "
    "The derived subquestions should be relevant and as generic as possible and should not be very specific to the user's specific scenario. "
    "If a question can't be derived, please fill n/a for question, otherwise proceed with classifying the question. "
    "The classification involves identifying the high level group and a topic within the group for each subquestion. "
    "The group and topic mapping is described below in `classification_criteria`. "
    "If any question can't be derived, please fill n/a for question, group and topic. "
    "Follow the style and tone of the example_questions specified below. Don't answer example questions. "
    "Please do not forcefit a question if the intent in the scenario is ambiguous and doesn't describe a question. \n\n"
    "example_questions: [ "
        "What do I need to do if a customer is getting an application loop?,"
        "How do I unlock an account?,"
        "What documents are needed to verify citizenship?,"
        "How much time does a consumer have to submit an ROP reinstatement request after notice?"
    "]"
    "classification_criteria:[ {{'group': 'Tech Aupport, 'topics: ['application updates', 'account creation', 'account unlock', 'password reset', 'account reclaim/access', 'ticket creation', 'consumer portal issues', 'Auth & DUO']}}, "
    "{{'group': 'DMI (Data Mismatch Issues)', 'topics':['income sources', 'medicare PDM (Periodic Data Matching', 'ROP(Reasonable Opportunity Period) - APTC (Advance Premium Tax Credit) issues', 'documentation mismatch' ]}}, "
    "{{'group': 'Eligibility', 'topics' : ['Medicare', 'Medicaid', 'Financial Assistance (APTC,CSR)', 'Qualified Health Plan (QHP)', 'Federal Tax Return (FTR)', 'Affordability rules and estimates', 'QLE/SEP']}}, "
    "{{'group': 'Account', 'topics' : ['Account Transfer', 'Application submission', 'Remote Identity Proofing (RIDP)', 'Income change', 'address change', 'demographic change']}}, "
    "{{'group': 'Enrollment assistance', 'topics': ['Reinstatement', 'Retroactive Voluntary termination/cancellation ', 'Prospective voluntary termination/cancellation', 'Financial assistance (APTC/CSR)', 'Coverage effective dates', 'Plan Selection', 'Plan Change', 'Binder payment', 'Enrollment finalizing', 'Net premium change', 'Enrollment Discrepancy with Carrier', 'Renewal', 'Id cards/billing payment']}}, "
    "{{'group': 'Miscellaneous', 'topics': [ '1095-A', 'Complaint', 'Appeal', 'Supervisor call request', 'Assister/Broker Training', 'Assister/Broker profile changes' , 'Assistant/Broker Designation', 'Assistant/Broker BOB']}}]\n\n"
    
    "scenario: {scenario}\n\n",
)