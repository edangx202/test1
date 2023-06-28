from langchain.prompts import PromptTemplate

prompt_template_questions = """
Your goal is to prepare a student for their an exam. You do this by asking questions about the text below:

{text}

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])


refine_template_questions = ("""
Your goal is to help a student prepare for an exam.
We have received some questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below
"------------\n"
"{text}\n"
"------------\n"

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions. Make sure to be detailed in your questions.
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)


# Prepare prompt to limit correct answer to be within 3 sentences, used by RetrievalQA
## Reference: https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
GENERATE_RIGHT_ANS = PromptTemplate.from_template("""
                                                Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                                                {context}
                                                Question: {question}
                                                Answer within 300 characters:
                                                """)


GENERATE_WRONG_ANS = PromptTemplate.from_template("""
                    You are a teacher preparing multiple-choice questions and answers for a test.

                    The question is defined in text delimited by triple hash.
                    The correct answer for the question is defined in text delimited by triple backticks.

                    The wrong answer should have similar number of characters as compared to the correct answer
                    The wrong answer should miss some facts from correct answer.
                    Do not apply negation or antonyms of words from the correct answer.

                    Proivde the response in the form of a numbered list.
                    Do not include the phrase "Possible wrong answers:"

                    ###{question}###
                    Generate 3 different wrong answers using the correct answer.
                    '''{correct_ans}'''
                    """)