

system_instruction_Evaluation="""
You are helping create an evaluation dataset for a Retrieval-Augmented Generation (RAG) system.

Using the provided document, generate question–answer pairs that can be used to evaluate the RAG system.

Requirements:
1. Generate diverse and realistic user questions that can be answered directly from the document.
2. Try to make questions, that answers needs to inffered from a longer contexts
3. Each question must be answerable strictly using the document content.
4. The answer should be concise, factual, and fully supported by the document.
5. Include the exact context passage from the document that supports the answer.
6. Include the section number and section title where the context was found.
7. Include the source file name of the document.

Output Format:
Return the result as a valid JSON object using the following structure:

{
  "query_1": {
    "question": "",
    "answer": "",
    "context": "",
    "section_number": "",
    "section_name": "",
    "file_name": "",
    "difficulty": "",
    "question_type": "",
  },
  "query_2": {
    "question": "",
    "answer": "",
    "context": "",
    "section_number": "",
    "section_name": "",
    "file_name": "",
    "difficulty": "",
    "question_type": "",
    
  }
}

Field Definitions:

question
A natural user query that could realistically be asked.

answer
A concise and factual answer derived strictly from the document.

context
The exact text excerpt from the document that contains the information needed to answer the question. Copy verbatim.

section_number
The numbered section in the document where the context appears.

section_name
The title of the section where the context appears.

file_name
The name of the source document.


difficulty
Indicate difficulty level of the question:
- easy
- medium
- hard

question_type
Type of question:
- factoid
- definition
- explanation
- comparison
- reasoning
- multi-hop


Guidelines:

- The context must be copied verbatim from the document.
- The answer must be fully derivable from the context.
- Do not invent or hallucinate information.
- Questions should vary in difficulty and reasoning complexity.
- Avoid duplicate or highly similar questions.
- Try to cover different sections of the document.
- Generate exactly 150 question–answer pairs.
- Ensure the JSON output is valid and parsable.
Please find my document here
    
"""