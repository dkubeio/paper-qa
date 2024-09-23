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
    "You don't need to directly answer the question yes or no. If it is a policy related question, explain the policy "
    "If the question is asking for a procedure, answer the process. "
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). \n"
    "Include confidence score of the generated answer on the scale of 1 to 10 \n"
    "Do not explain Confidence score. Do not generate anything after generating confidence score. \n"
    "Context:\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
)

#"If the context doesn't provide answer, but provides policy for a part of the question, state the policy. "

qa_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template= "Your job is to extract relevant context for the user's question. "
    "Never directly answer yes or no, but only provide policy or procedural information from relevant sections "
    "Do not assume anything. Use the context and not any prior learnings. "
    "Please limit the output to 100 words. "
    "Please do not include any explanatory logic or notes. "
    "For each part of your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). "
    "Note that pregnancy is not a QLE, but child birth is a QLE. \n"
    "Include sources used at the end of the response"
    "Include confidence score of the generated summary on the scale of 1 to 10 \n"
    "Do not explain Confidence score or Context. Do not generate anything after generating confidence score.  \n"
    "If the context provides insufficient information reply `I cannot answer, Please escalate to supervisor or rephrase the question` and don't provide any further information. "
    "If the context provides sufficient information reply strictly in the format; Answer: ...\n Sources: ...\n Confidence score: ... "
    "Context:\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
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

system_prompts = {
    'General': "You are a Retrieval Augmented Generation chatbot. Think step by step and answer in a direct and concise tone. ",
    'NV' : "You are an expert Call Center Agent Assist in the public healthcare insurance marketplace, NVHL, for the state of Nevada. Think step by step and answer in a direct and concise tone.\n",
    'PA' : "You are an expert Call Center Agent Assist in the public healthcare insurance marketplace, Pennie, for the state of Pennsylvania. Think step by step and answer in a direct and concise tone.\n",
    'VA' : "You are an expert Call Center Agent Assist in the public healthcare insurance marketplace, Virginia Health Insurance Marketplace, for the state of Virginia. Think step by step and answer in a direct and concise tone.\n",
    'MO' : "You are an expert Child Welfare Agent Assist in Missouri Department of Social Services, DSS. Think step by step and answer in a direct and concise tone.\n",
    'GA' : "You are an expert Policy and Manual Management System (PAMMS) Agent Assist in Division of Family and Children Services, DFCS in state of Georgia. Think step by step and answer in a direct and concise tone.\n",
    "UI": "You are an expert Policy and Manual Management System Agent Assist in the Unemployment Insurance Division. Think step by step and respond in a direct and concise tone. \n",
    "compare_qa": "You are an expert question analyzer. Think step by step and respond in a direct and concise tone. \n",
}

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

csr_rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    #template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent which should also include a policy and procedure question, and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "
    #template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent of the scenario, and suggest upto 4 additional followup questions. Classify each question using the classification_criteria provided.  "
    #        "Use the following guidelines for the task. \n"
    template="Your task is to analyze the customer scenario. "
            "If the scenario is a well formed question, fix the typos if any and derive upto 4 followup questions. "
            "Derive meaningful questions without changing the intent. "
            "Suggest upto 4 followup questions as well. "
            "Use the example_questions below to determine if the scenario is a well formed question. "
            "Classify each question using the classification_criteria provided.  "
            "Use the following guidelines for the task. \n"
            "- scenario starting with How, What, Can, Will, Why would typically tend to be well formed question. "
            "- Policy questions start with 'What is the policy' and procedural questions start with 'How to'. If the scenario is not a well formed question, generate both the variants for each derived question. \n "
            "- Questions that have the action words are typically procedural questions. \n"
            "- list each question separately as specified in the json format {json_format} and combine them into a single json list. \n"
            # "- if the original `scenario` describes a Policy, place the Policy questions at the top of the json list, but if it describes a procedure, place procedural question at the top of the json list. \n "
            "- Ensure the output is strictly formatted as a JSON list without any additional text, explanations, or notes. \n"
            "- Each question must have one group, one topic, and a similarity_score (1-10). Use the classification_criteria below to determine the appropriate group and topic. Do not invent new groups or topics. \n"
            "- similarity_score describes how similar is the generated question to the scenario. \n"
            "- If no meaningful question can be derived, return n/a for the question. \n"
            "- Follow the style and tone of the provided example_questions. \n"
            "- Ensure questions capture the user's intent and include any specific error/warning messages mentioned. \n" 
            "- Retain acronyms exactly as given in the scenario. \n"
            "- Retain the nouns that are related to Affordable Care Act (ACA) and Insurance Marketplace exactly as given in the scenario. \n"
            "- If the `scenario` describes a time period (such as a month, a year etc), you must ensure the time period is copied to the derived questions. \n"
            "- Do not forcefit a question if the scenario's intent is ambiguous or doesn't describe a question. \n"
            "- If the scenrio is about no income benefits, generate free and low-cost benefits questions as well. \n"
            "- A reference to a number in the scenario could mean contact number. \n"
            "- when ever the words cx/client/my is used, it should be replaced with customer. \n"
            "- Its always the coverage that is terminated, not the customer. if terminated is used with customer, rephrase it with coverage\n"
            "- If the word backdate or back date is used, it should be replaced with effective date change \n"
            "- Remove any personal information such as names, IDs, address from the scenario. \n\n"

    "example_questions: "
        "  [ 'What should I do when APTCs were not applied to a month due to Medicaid termination?', "
        " 'Why did my Advanced Premium Tax Credit disappear?', "
        # " 'Can self-employed individuals get Penny coverage?', "
        " 'Will I receive a 1095 form if I am enrolled in Medicaid?', "
        # " 'I was denied Medicaid, but I have no income, can I apply for Pennie?',"
        " 'Why isn't pregnancy considered a Qualifying Life Event (QLE)?', "
        " 'What is the contact number for Medicaid inquiries?', "
        " 'What is the policy for effective date change request', "
        " 'What should I do if an AOR calls about a ticket?', "
        " 'How do I get status of a ticket?', "
        " 'Why was my client's coverage terminated?', "
        " 'What do I need to do if a customer is getting an application loop?', "
        " 'How do I unlock an account?', "
        " 'What is Medicaid contact number?', "
        " 'What documents are needed to verify citizenship?', "
        " 'How much time does a consumer have to submit an ROP reinstatement request after notice?', "
        " 'What should I do if my plan was terminated after removing my husband from enrollment?', "
        " 'How do I provide ticket status to a consumer?' ] \n\n"

    "classification_criteria: "
        "[ {{'group': 'Tech Aupport, 'topics': ['application updates', 'account creation', 'account unlock', 'password reset', 'account reclaim/access', 'ticket creation', 'consumer portal issues', 'Auth & DUO']}}," 
        "  {{'group': 'DMI (Data Mismatch Issues)', 'topics':['income sources', 'medicare PDM (Periodic Data Matching', 'ROP(Reasonable Opportunity Period) - APTC (Advance Premium Tax Credit) issues', 'documentation mismatch', 'Medicaid' ]}}," 
        "  {{'group': 'Eligibility', 'topics' : ['Medicare', 'Medicaid', 'Financial Assistance (APTC,CSR)', 'Qualified Health Plan (QHP)', 'Federal Tax Return (FTR)', 'Affordability rules and estimates', 'QLE/SEP', 'residency']}}," 
        "  {{'group': 'Account', 'topics' : ['Account Transfer', 'Application submission', 'Remote Identity Proofing (RIDP)', 'Income change', 'address change', 'demographic change', 'payments']}}," 
        "  {{'group': 'Enrollment assistance', 'topics': ['Reinstatement', 'Retroactive Voluntary termination/cancellation ', 'Prospective voluntary termination/cancellation', 'Financial assistance (APTC/CSR)', 'Coverage effective dates', 'Plan Selection', 'Plan Change', 'Binder payment', 'Enrollment finalizing', 'Net premium change', 'Enrollment Discrepancy with Carrier', 'Renewal', 'Id cards/billing payment']}}," 
        "  {{'group': 'Miscellaneous', 'topics': [ '1095-A', 'Complaint', 'Appeal', 'Supervisor call request', 'Assister/Broker Training', 'Assister/Broker profile changes' , 'Assistant/Broker Designation', 'Assistant/Broker BOB']}}]\n\n"
        " \n\n"
        "ambiguous_scenarios: ['calling ticket', 'income', 'escalate', 'insurance']\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"What is the status of ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}}]\n"
        "Or,\n"
        "[{{\"question\": \"How to check the status of Ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n"
    "Correct format:\n"
        "[{{\"question"": \"What does policy states about the status of a ticket?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}},"
          "{{\"question\": \"How to check the status of a Ticket?\", \"group"": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B for Smith Li?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"Do we have medicaid number on file?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"Medicaid\", \"confidence_score\": 9}},"
         "{{\"question\": \"Why isn't my SEP functioning?\", \"group\": \"Enrollment assistance\", \"topic\": \"SEP\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of TIC-123456's income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "Correct format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the contact number for Medicaid inquiries?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"Medicaid\", \"confidence_score\": 9}},"
         "{{\"question\": \"What are the reasons for not being able to avail my SEP?\", \"group\": \"Enrollment assistance\", \"topic\": \"SEP\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "\n\n"
    "Scenario: {scenario} ",
)

csr_rewrite_prompt_raw = PromptTemplate(
    input_variables=["scenario", "json_format"],
    #template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent which should also include a policy and procedure question, and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "
    template="Your task is to fix only the typos in the attached question without changing the intent of the question. Do not add/remove anything to the question. Classify the question using the classification_criteria provided and provide upto 4 additional followup questions.  "
            "Use the following guidelines for the task. \n"
            "- list each question separately as specified in the json format {json_format} and combine them into a single json list. \n"
            "- Ensure the output is strictly formatted as a JSON list without any additional text, explanations, or notes. \n"
            "- Each question must have one group, one topic, and a similarity_score (1-10). Use the classification_criteria below to determine the appropriate group and topic. Do not invent new groups or topics. \n"
            "-- similarity_score describes how similar is the generated question to the scenario. \n"
            #"- If no meaningful question can be derived, return n/a for the question. \n"
            #"- Follow the style and tone of the provided example_questions for followup questions only. \n"
            #"- Ensure followup questions capture the user's intent and include any specific error/warning messages mentioned. \n" 
            "- Retain acronyms exactly as given in the scenario. \n"
            #"- If the `scenario` describes a time period (such as a month, a year etc), you must ensure the time period is copied to the derived questions. \n"
            #"- Do not forcefit a question if the scenario's intent is ambiguous or doesn't describe a question. \n"
            "- If the question is about no income benefits, generate free and low-cost benefits followup questions as well. \n"
            "- A reference to a number in the question could mean contact number. \n"
            "- Remove any personal information such as names, IDs, address from the question. \n\n"

    #"example_questions: \n"
    #    "  [ 'What should I do when APTCs were not applied to a month due to Medicaid termination?', 'I was denied Medicaid, but I have no income, can I apply for Pennie?', 'Why is not pregnancy considered a Qualifying Life Event (QLE)?', 'What is the contact number for Medicaid inquiries?', 'What is the policy for effective date change request', 'What should I do if an AOR calls about a ticket?', 'How do I get status of a ticket?`, 'What do I need to do if a customer is getting an application loop?','How do I unlock an account?','What documents are needed to verify citizenship?','How much time does a consumer have to submit an ROP reinstatement request after notice?', 'How do I provide ticket status to a consumer?']\n\n"
    #    "\n\n"
    "classification_criteria: \n"
        "[ {{'group': 'Tech Aupport, 'topics': ['application updates', 'account creation', 'account unlock', 'password reset', 'account reclaim/access', 'ticket creation', 'consumer portal issues', 'Auth & DUO']}}," 
        "  {{'group': 'DMI (Data Mismatch Issues)', 'topics':['income sources', 'medicare PDM (Periodic Data Matching', 'ROP(Reasonable Opportunity Period) - APTC (Advance Premium Tax Credit) issues', 'documentation mismatch', 'Medicaid' ]}}," 
        "  {{'group': 'Eligibility', 'topics' : ['Medicare', 'Medicaid', 'Financial Assistance (APTC,CSR)', 'Qualified Health Plan (QHP)', 'Federal Tax Return (FTR)', 'Affordability rules and estimates', 'QLE/SEP', 'residency']}}," 
        "  {{'group': 'Account', 'topics' : ['Account Transfer', 'Application submission', 'Remote Identity Proofing (RIDP)', 'Income change', 'address change', 'demographic change', 'payments']}}," 
        "  {{'group': 'Enrollment assistance', 'topics': ['Reinstatement', 'Retroactive Voluntary termination/cancellation ', 'Prospective voluntary termination/cancellation', 'Financial assistance (APTC/CSR)', 'Coverage effective dates', 'Plan Selection', 'Plan Change', 'Binder payment', 'Enrollment finalizing', 'Net premium change', 'Enrollment Discrepancy with Carrier', 'Renewal', 'Id cards/billing payment']}}," 
        "  {{'group': 'Miscellaneous', 'topics': [ '1095-A', 'Complaint', 'Appeal', 'Supervisor call request', 'Assister/Broker Training', 'Assister/Broker profile changes' , 'Assistant/Broker Designation', 'Assistant/Broker BOB']}}]\n\n"
        " \n\n"
        "ambiguous_scenarios: ['calling ticket', 'income', 'escalate', 'insurance']\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"What is the status of ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}}]\n"
        "Or,\n"
        "[{{\"question\": \"How to check the status of Ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n"
    "Correct format:\n"
        "[{{\"question"": \"What does policy states about the status of a ticket?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}},"
          "{{\"question\": \"How to check the status of a Ticket?\", \"group"": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B for Smith Li?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"Do we have medicaid number on file?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"Medicaid\", \"confidence_score\": 9}},"
         "{{\"question\": \"Why isn't my SEP functioning?\", \"group\": \"Enrollment assistance\", \"topic\": \"SEP\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of TIC-123456's income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "Correct format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the contact number for Medicaid inquiries?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"Medicaid\", \"confidence_score\": 9}},"
         "{{\"question\": \"What are the reasons for not being able to avail my SEP?\", \"group\": \"Enrollment assistance\", \"topic\": \"SEP\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "\n\n"
    "Question: {scenario}\n\n",
)
   # "Your task is to analyze a customer scenario, derive a potential policy and procedure question if any, and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "

# Child welfare - Department of Social Services
dss_rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    #template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent which should also include a policy and procedure question, and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "
    #        "Use the following guidelines. \n"
    template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent. "
            "If the scenario is a well formed question, only fix the typos if any. Otherwise, derive policy and procedure questions, "
            "and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "
            "Use the following guidelines. \n"
            "- Policy questions start with 'What is the policy' and procedural questions start with 'How to'. Generate both the variants for each derived question.\n "
            "- If the scenario describes a question in a meaningful way, use the scenario as is in a question as well. \n"
            "- list each question separately as specified in the json format {json_format} and combine them into a single json list. \n"
            "- Ensure the output is strictly formatted as a JSON list without any additional text, explanations, or notes. \n"
            "- Each question must have one group, one topic, and a confidence score (1-10). Use the classification_criteria below to determine the appropriate group and topic. Invent new groups or topics as needed. Ensure consistency. \n"
            "- If no meaningful question can be derived, return n/a for the question. \n"
            "- Ensure questions capture the user's intent and include any specific error/warning messages mentioned. \n" 
            "- Retain acronyms exactly as given in the scenario. \n"
            "- Do not forcefit a question if the scenario's intent is ambiguous or doesn't describe a question. \n"
            "- Remove any personal information such as names, IDs from the scenario. \n\n"


    "classification_criteria: \n"
        "[ {{'group': 'Missouri Practice Model, 'topics': ['Family engagement', 'Safety', 'Questioning' , 'Safety and Risk Assessment', 'Engaging', 'Safety', 'Planning', 'Closure']}}," 
        "  {{'group': 'Intake', 'topics':['Reporting', 'Abuse', 'Referrals', 'Court' ]}}," 
        "  {{'group': 'Delivery of Services/Intact Families (FCS)', 'topics' : ['Case Opening', 'Case Planning', 'Case Monitoring', 'Case closure']}}," 
        "  {{'group': 'Alternative Care', 'topics' : ['Placements', 'Court', 'Support Teams', 'Adoption', 'Financials']}}," 
        "  {{'group': 'Case Record Maintenance and Access', 'topics': [ 'Filing', 'Documentation', 'Access', 'Transfer', 'Rentention and Expungement']}}," 
        "  {{'group': 'Resource Development', 'topics': [ 'Recruitment', 'Training', 'Emergency', 'Foster care']}}]\n\n"
        " \n\n"
 
    "Wrong format:\n"
        "[{{\"question\": \"What is the status of ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}}]\n"
        "Or,\n"
        "[{{\"question\": \"How to check the status of Ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n"
    "Correct format:\n"
        "[{{\"question"": \"What does policy states about the status of a ticket?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}},"
          "{{\"question\": \"How to check the status of a Ticket?\", \"group"": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B for Smith Li?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of TIC-123456's income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "Correct format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "\n\n"
    "Scenario: {scenario}\n\n",
)

ga_rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    template="Your task is to analyze the customer scenario, derive meanigful questions without changing the intent which should also include a policy and procedure question, and suggest upto 4 followup questions. Classify each question using the classification_criteria provided.  "
            "Use the following guidelines. \n"
            "- Policy questions start with 'What is the policy' and procedural questions start with 'How to'. Generate both the variants for each derived question.\n "
            "- If the scenario describes a question in a meaningful way, use the scenario as is in a question as well. \n"
            "- list each question separately as specified in the json format {json_format} and combine them into a single json list. \n"
            "- Ensure the output is strictly formatted as a JSON list without any additional text, explanations, or notes. \n"
            "- Each question must have one group, one topic, and a confidence score (1-10). Use the classification_criteria below to determine the appropriate group and topic. Invent new groups or topics as needed. Ensure consistency. \n"
            "- If no meaningful question can be derived, return n/a for the question. \n"
            "- Ensure questions capture the user's intent and include any specific error/warning messages mentioned. \n" 
            "- Retain acronyms exactly as given in the scenario. \n"
            "- Do not forcefit a question if the scenario's intent is ambiguous or doesn't describe a question. \n"
            "- Remove any personal information such as names, IDs from the scenario. \n\n"


    "classification_criteria: \n"
        "[{{ 'group':'Legal Framework','topics': ['ADA Overview','Section 504 Summary','Title II & III Details','Compliance Standards','Key Legal Cases' ] }},"
        "{{'group':'Accessibility','topics': ['Facility Access','Service Modifications','Communication Aids','Coordinator Role','Public Notices' ] }},"
        "{{'group':'Disability Criteria','topics': ['Disability Definition','Qualified Individuals','Exclusions','Mobility Aids','Service Animals' ] }},"
        "{{'group':'Responsibilities','topics': ['DFCS Obligations','Provider Compliance','Coordinator Duties','Staff Training','Self-Assessment' ] }},"
        "{{'group':'Public Communication','topics': ['Rights Notifications','Disability Interaction','Modification Requests','Complaint Process','Alternative Formats' ] }},"
        "{{'group':'Compliance','topics': ['Nondiscrimination','Reasonable Modifications','Data Collection','Training','Public Notifications' ] }},"
        "{{'group':'Rights','topics': ['Filing Complaints','Communication Assistance','Privacy Protection','LEP Services','Disability Accommodations' ] }},"
        "{{'group':'Programs','topics': ['SNAP','CSFP','TEFAP','USDA Compliance','HHS Compliance' ] }},"
        "{{'group':'Responsibilities','topics': ['Staff Obligations','Contractor Duties','Reporting Noncompliance','Monitoring Procedures','Review Processes' ] }},"
        "{{'group':'Legal Framework','topics': ['Civil Rights Act','Title VI','Title IX','Rehabilitation Act','Federal Guidelines' ] }},"
        "{{'group':'Voter Registration','topics': ['Background':'NVRA Overview','Requirements':'Document Distribution','Procedures':'Forms and Processing','Customer Assistance':'Voter Registration Support','Confidentiality':'Information and Records','Contacts':'Getting Help'] }},"
        "{{'group':'CSBG','topics': ['Program Overview','Board Governance','Needs Assessment','Action Plan','Fiscal Management','Types of Income','Client Eligibility','ADA & Section 504','Fair Hearing' ] }},"
        "{{'group':'LIHEAP','topics': ['Program Overview','Program Authorization','Fraud Prevention','Vendor Management','Weatherization','Program Monitoring','Fair Hearing','ADA & Section 504' ] }},"
        "{{'group':'Medicaid','topics':['2000':'General Information','2050':'Application Processing','2100':'Classes of Assistance','2200':'Eligibility Criteria','2300':'Resources','2400':'Income','2500':'ABD Budgeting','2550':'Patient Liability/Cost Share','2575':'Nursing Home Payments','2600':'Family Assistance Units','2650':'Family Budgeting','2700':'Case Management','2800':'Children in Placement','2900':'Referrals','ABD Financial Limits','Family Financial Limits']}},"
        "{{'group':'SNAPProgram Overview','topics': ['General Program Overview','Application Process Overview','Assistance Units Overview','Basic Eligibility Criteria Overview','Financial Eligibility Criteria Overview' ] }},"
        "{{'group':'SNAP System Processes','topics': ['Computer Matches Overview','Budgeting Overview','Ongoing Case Management Overview','Issuance Overview']}},"
        "{{'group':'SNAP Miscellaneous','topics': ['Financial Standards','Hearings','Manual Transmittal Cover Letters','Case Record Maintenance and Document Management','Glossary','Forms','Customer Complaint Procedures','Child Support Services Glossary' ] }}] \n\n"
 
    "Wrong format:\n"
        "[{{\"question\": \"What is the status of ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}}]\n"
        "Or,\n"
        "[{{\"question\": \"How to check the status of Ticket #-123456?\", \"group\": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n"
    "Correct format:\n"
        "[{{\"question"": \"What does policy states about the status of a ticket?\", \"group\": \"Tech Support\", \"topic\": \"ticket creation\", \"confidence_score\": 10}},"
          "{{\"question\": \"How to check the status of a Ticket?\", \"group"": \"Tech Support\", \"topic\": \"ticket status\", \"confidence_score\": 10}}]\n\n"
    "Wrong format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B for Smith Li?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of TIC-123456's income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "Correct format:\n"
        "[{{\"question\": \"How to apply for Medicare Part B?\", \"group\": \"Enrollment assistance\", \"topic\": \"Plan Selection\", \"confidence_score\": 9}},"
         "{{\"question\": \"What is the reason for the rejection of income change?\", \"group\": \"DMI (Data Mismatch Issues)\", \"topic\": \"income change\", \"confidence_score\": 9}}]\n"
    "\n\n"
    "Scenario: {scenario}\n\n",
)

ui_rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    template="Your task is to analyze the unemployment insurance-related customer scenario, derive meaningful questions without changing the intent which should also include a policy and procedure question, and suggest up to 4 follow-up questions. Classify each question using the classification_criteria provided. \n"
            "Use the following guidelines:\n"
            "- Policy questions start with 'What is the policy' and procedural questions start with 'How to'. Generate both the variants for each derived question.\n"
            "- If the scenario describes a question in a meaningful way, use the scenario as is in a question as well.\n"
            "- List each question separately as specified in the JSON format {json_format} and combine them into a single JSON list.\n"
            "- Ensure the output is strictly formatted as a JSON list without any additional text, explanations, or notes.\n"
            "- Each question must have one group, one topic, and a confidence score (1-10). Use the classification_criteria below to determine the appropriate group and topic. Invent new groups or topics as needed. Ensure consistency.\n"
            "- If no meaningful question can be derived, return n/a for the question.\n"
            "- Ensure questions capture the user's intent and include any specific error/warning messages mentioned.\n"
            "- Retain acronyms exactly as given in the scenario.\n"
            "- Do not forcefit a question if the scenario's intent is ambiguous or doesn't describe a question.\n"
            "- Remove any personal information such as names, IDs from the scenario.\n\n"

    "classification_criteria: \n"
        "[{{'group':'Claims and Benefits','topics':['Filing a Claim','Weekly Certifications','Monetary Eligibility','Non-Monetary Eligibility','Benefit Payments','Overpayment and Recovery'] }},"
        "{{'group':'Eligibility and Requirements','topics':['Job Search Requirements','Able and Available to Work','Separation Issues','Work Search Audits','Job Training Programs','Appeals Process'] }},"
        "{{'group':'Program Policies','topics':['UI Policy Overview','Federal Guidelines','State-Specific Regulations','Benefit Year Provisions','Extended Benefits Programs','Employer Contributions'] }},"
        "{{'group':'Application Process','topics':['Online Claim Filing','Claim Status Inquiries','Document Submission','Identity Verification','Error Messages and Troubleshooting','Customer Support'] }},"
        "{{'group':'Fraud Prevention and Security','topics':['Identity Theft Prevention','Fraud Reporting','Overpayment Fraud','System Security','Protecting Personal Information','Unauthorized Claims'] }},"
        "{{'group':'Payments and Financials','topics':['Payment Methods','Direct Deposit Setup','Tax Withholding','Payment Delays','Payment Corrections','Tax Documents'] }},"
        "{{'group':'Workforce Programs','topics':['Reemployment Services','Job Training Programs','Work Search Assistance','Career Counseling','Job Placement'] }},"
        "{{'group':'Employer Responsibilities','topics':['Reporting New Hires','Quarterly Wage Reporting','Employer Contribution Rates','Appeals Against Claims','Workplace Separation Policies','Shared Work Programs'] }},"
        "{{'group':'Technical Support','topics':['Website Navigation','Account Login Issues','System Errors','Document Upload Issues','Online Chat Assistance'] }}]\n\n"

    "\n\n"
    "Scenario: {scenario}\n\n",
)

   
'''
    "You are an expert Call Center Agent Assist in the public healthcare insurance marketplace. "
    "Your job is to analyze a customer scenario, derive all the policy or procedure subquestions and classify the questions based on some criteria below. "
    "list each question separately as specified in the json format {json_format} and combine them into a single json list. "
    "Each element in the json list should be a json object with 3 key/value pairs, where the keys are question, group and topic. "
    "Strictly follow this format and do not include any other text, explanation or notes "
    "If the scenario is ambiguous and doesn't describe any question, please return n/a for question, otherwise proceed with the classification of question. "
    "The derived subquestions should be relevant and as generic as possible and should not be very specific to the user's specific scenario. "
    "Copy the expansion for acronyms from the scenario. You must not expand from your memory"
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

'''

rewrite_prompts = {
    'NV' : csr_rewrite_prompt,
    'PA' : csr_rewrite_prompt,
    'PA_raw' : csr_rewrite_prompt_raw,
    'NV_raw' : csr_rewrite_prompt_raw,
    'MO' : dss_rewrite_prompt,
    'GA' : ga_rewrite_prompt,
    'VA' : csr_rewrite_prompt,
    'UI' : ui_rewrite_prompt,
}


#"Respond in a JSON format as specified below {json_format}. Strictly follow this format and do not include any other text: "
#   "where question is derived from the scenario, group and topic are derived by classifying the derived question. "
rewrite_prompt = PromptTemplate(
    input_variables=["scenario", "json_format"],
    template="You are an expert Call Center Agent Assist in the public healthcare insurance marketplace. "
    "Your job is to analyze a customer scenario, derive all the policy or procedure subquestions and classify the questions based on some criteria below. "
    "list each question separately as specified in the json format {json_format} and combine them into a single json list. "
    "Each element in the json list should be a json object with 3 key/value pairs, where the keys are question, group and topic. "
    "Strictly follow this format and do not include any other text, explanation or notes "
    "If the scenario is ambiguous and doesn't describe any question, please return n/a for question, otherwise proceed with the classification of question. "
    "The derived subquestions should be relevant and as generic as possible and should not be very specific to the user's specific scenario. "
    "Copy the expansion for acronyms from the scenario. You must not expand from your memory"
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


compare_question_prompt = PromptTemplate( 
    input_variables=["question_1", "question_2"],
    template="""
    Compare the following two questions and rate their similarity on a scale of 1-10, where:

    1 indicates they are completely different in meaning and context,
    10 indicates they are almost identical or fully similar in meaning and context.

    Consider the following aspects in your rating:

    Do both questions seek the same or very similar information?
    Are the contexts of both questions aligned (e.g., same topic, subject matter)?
    Is the phrasing similar, even if expressed differently?
    How closely do the intents of the two questions match?

    Strictly give score in this format 'Score: [[score]]/10' for example 'Score: 8/10'

    Question 1:
    {question_1}

    Question 2:
    {question_2}
    Response: 
    """
    )
