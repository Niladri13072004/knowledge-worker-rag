from streamlit.proto import openmetrics_data_model_pb2
import traceback
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import Docx2txtLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import json
import plotly.express as px
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import traceback
import re
import math
import time
import shutil


load_dotenv()
parent_docs = {}
chat_history = []
pending_clarification = None
conversation_summary = ""

MODEL_NAME = "openai/gpt-oss-20b:free"

ENABLE_REWRITING = False
ENABLE_SELF_REFLECTION = False
ENABLE_QUERY_REWRITE = True
ENABLE_SELF_CORRECTION = False
ENABLE_GENERAL_ROUTING = False
is_subquestion = False

ENABLE_MULTI_QUERY = True
ENABLE_RERANKING = True
ENABLE_PARENT_RETRIEVAL = True
ENABLE_CONTEXT_COMPRESSION = True
ENABLE_HUMAN_CLARIFICATION = False
ENABLE_ANSWER_VERIFICATION = False

judge_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:7860",
        "X-OpenRouter-Title": "Knowledge Worker Judge",
    },
)

judge_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

local_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_answer(prompt):

    print("\nOPENROUTER CALL")
    print(prompt[:100])

    for attempt in range(5):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            if response is None:
                raise ValueError(
                    "OpenRouter returned None"
                )

            if response.choices is None:
                raise ValueError(
                    "OpenRouter returned no choices"
                )

            if len(response.choices) == 0:
                raise ValueError(
                    "OpenRouter returned empty choices"
                )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if not content:
                raise ValueError(
                    "OpenRouter returned empty content"
                )

            return content

        except Exception as e:

            error_text = str(e)

            print("\nOPENROUTER ERROR:")
            print(error_text)

            if "429" in error_text:

                print(
                    "Rate limited. "
                    "Waiting 30 seconds..."
                )

                time.sleep(30)

                continue

            raise

    raise RuntimeError(
        "OpenRouter failed after 5 retries"
    )

def answer_general_question(question):

    prompt = f"""
    Answer the following question.

    Question:
    {question}
    """

    return generate_answer(
        prompt
    )

def self_evaluate(question,context,answer):

    prompt = f"""
    You are evaluating a RAG answer.

        Question:
        {question}

        Context:
        {context}

        Answer:
        {answer}

    Evaluate:

        1. Is answer supported by context?
        2. Is answer complete?
        3. Is answer relevant?
        4. Confidence score (0-10)

        Return ONLY JSON:

        {{
            "supported": true,
            "complete": true,
            "confidence": 8,
            "feedback": "short explanation"
        }}
    """

    try:

        response = judge_llm.invoke(prompt)
        response = response.content
        print("\nRAW SELF EVAL")
        print(response)
        return _extract_json_object(
            response
        )

    except Exception as e:

        print(
            "\nSELF EVAL ERROR:"
        )

        print(str(e))

        return {
            "supported": True,
            "complete": True,
            "confidence": 10,
            "feedback": ""
        }

def self_correct_answer(
    question,
    context,
    answer,
    feedback
):

    prompt = f"""
    You are improving an answer.

    Question:
    {question}

    Context:
    {context}

    Current Answer:
    {answer}

    Review Feedback:
    {feedback}

    Improve the answer using ONLY the context.

    Rules:
    - Keep correct information.
    - Fix inaccuracies.
    - Add missing information.
    - Preserve exact numbers.
    - Do not hallucinate.

    Return only the improved answer.
    """

    response = judge_llm.invoke(prompt)

    return response.content.strip()

def verify_answer(
    question,
    context,
    answer
):

    prompt = f"""
You are a fact checking system.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON.

{{
    "supported": true,
    "confidence": 10,
    "feedback": "short explanation"
}}
"""

    try:

        response = judge_llm.invoke(
            prompt
        )

        print("\nRAW VERIFY")
        print(response.content)

        return _extract_json_object(
            response.content
        )

    except Exception as e:

        print(
            "\nVERIFY ERROR:"
        )

        print(str(e))

        return {
            "supported": True,
            "confidence": 10,
            "feedback": "Verification failed"
        }

def improve_answer(
    question,
    context,
    answer,
    feedback
):

    prompt = f"""
Question:
{question}

Context:
{context}

Previous Answer:
{answer}

Feedback:
{feedback}

Generate a better answer.

Use only the context.
"""

    return generate_answer(
        prompt
    )

splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def rewrite_chunk(text):

    prompt = f"""
    Rewrite the following text for retrieval.

    Rules:
    1. Preserve all facts.
    2. Add clarifying context.
    3. Expand abbreviations.
    4. Add searchable keywords.
    5. Do NOT invent information.

    Text:
    {text}
    """

    return generate_answer(prompt)

knowledge_bases = {}
vectordb = None
retriever = None
bm25 = None
all_chunks = None

def process_document(files):
    global vectordb
    global retriever
    global bm25
    global all_chunks
    global parent_docs
    global conversation_summary

    print("process_document called")

    try:
        vectordb = None
        retriever = None
        bm25 = None
        all_chunks = None

        parent_docs.clear()
        chat_history.clear()
        pending_clarification = None
        conversation_summary = ""

        if not files:
            return "No files uploaded."

        all_docs = []
        parent_id = 0

        for file in files:
            file_name = file.name.lower()

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file.name)
            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["filename"] = os.path.basename(file.name)
                doc.metadata["parent_id"] = parent_id
                parent_docs[parent_id] = doc.page_content
                parent_id += 1

            all_docs.extend(docs)

        if not all_docs:
            return "No supported content found in the uploaded files."

        chunks = splitter.split_documents(all_docs)

        for idx, chunk in enumerate(chunks):
            if "filename" not in chunk.metadata:
                chunk.metadata["filename"] = chunk.metadata.get("source", "unknown")

            if "document_name" not in chunk.metadata:
                chunk.metadata["document_name"] = chunk.metadata.get("filename", "unknown")

            if "parent_id" not in chunk.metadata:
                # fall back to a stable id if the splitter did not preserve metadata
                chunk.metadata["parent_id"] = idx

        if ENABLE_REWRITING:
            print("Rewriting Chunks...")
            for chunk in chunks:
                try:
                    rewritten = rewrite_chunk(chunk.page_content)
                    if not rewritten:
                        continue

                    chunk.metadata["original_text"] = chunk.page_content
                    chunk.metadata["retrieval_text"] = rewritten
                    chunk.page_content = rewritten

                except Exception:
                    print("\nREWRITE ERROR")
                    traceback.print_exc()

        all_chunks = chunks

        if not chunks:
            return "No chunks were created from the uploaded files."

        tokenized_corpus = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        print("Total Chunks Created:", len(chunks))

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
        )

        knowledge_bases["default"] = vectordb

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5}
        )

        print("retriever created")
        return f"""
        Documents Loaded: {len(files)}

        Chunks Created: {len(chunks)}

        Knowledge Base Ready
        """

    except Exception:
        error = traceback.format_exc()
        print(error)
        return error

def chat(message, history):
    global retriever

    if retriever is None:
        history.append(
            {
                "role": "assistant",
                "content": "Please upload documents first."
            }
        )
        return history, history, ""

    try:
        start = time.time()
        print("CALLING answer_question")
        answer, combined_docs = answer_question(message)
        print("ANSWER RECEIVED")
        sources = []
        for doc in combined_docs:
            filename = doc.metadata.get("filename", "Unknown File")
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"{filename} - Page {page + 1}")

        source_text = "\n".join(sorted(set(sources)))
        print("CHAT RETURNING")
        final_answer = answer
        if source_text:
            final_answer = f"{answer}\n\nSources:\n{source_text}"

        print(f"LLM Time: {time.time() - start:.2f} sec")

        history.append(
            {
                "role": "user",
                "content": message
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )

        return history, history, ""

    except Exception as e:
        history.append(
            {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            }
        )
        return history, history, ""

def classify_query(
    question
):

    prompt = f"""
Classify:

{question}

Choose one:

definition
concept
numerical
spanning
holistic
reasoning

Question:
What is the primary function of a compressor?

definition

Question:
Why is the compressor called the heart of a refrigeration system?

reasoning

Question:
What is multistage compression?

concept

Question:
What was Niladri's CGPA?

numerical

Return only label.
"""

    response = (
        local_llm.invoke(
            prompt
        )
    )

    return (
        response.content
        .strip()
        .lower()
    )

def classify_intent(
    question
):

    prompt = f"""
Classify the intent.

Question:
{question}

Choose one:

howto
comparison
troubleshooting
definition
find_doc
summarization
fact_lookup
analysis

Return only label.
"""

    response = local_llm.invoke(
        prompt
    )

    return (
        response.content
        .strip()
        .lower()
    )

def classify_question_source(question):

    prompt = f"""
You are routing questions.

IMPORTANT:

If the question could be answered
from uploaded documents,
choose DOCUMENT.

Choose GENERAL only when
the question clearly requires
outside knowledge and not
document contents.

Choose HYBRID if both are needed.

Question:
{question}

Return only:

DOCUMENT
GENERAL
HYBRID
"""

    response = (
        local_llm.invoke(
            prompt
        )
    )

    return (
        response.content
        .strip()
        .upper()
    )

def detect_ambiguity(question):

    prompt = f"""
Determine whether the question truly requires clarification.

Only mark ambiguous if it CANNOT be answered.

Examples:

Question:
Can you provide a summary of Niladri's technical projects?

Answer:
{{
  "ambiguous": false,
  "clarification": ""
}}

Question:
What percentage did Niladri score in Higher Secondary?

Answer:
{{
  "ambiguous": false,
  "clarification": ""
}}

Question:
How do I update it?

Answer:
{{
  "ambiguous": true,
  "clarification": "What does 'it' refer to?"
}}

Question:
Compare them.

Answer:
{{
  "ambiguous": true,
  "clarification": "Which items would you like compared?"
}}

Question:
Which skills listed in the resume were applied in the Conversational RAG Knowledge Assistant project?

Answer:
{{
  "ambiguous": false,
  "clarification": ""
}}

Question:
Can you provide a summary of Niladri's technical projects?

Answer:
{{
  "ambiguous": false,
  "clarification": ""
}}

Question:
{question}
"""

    try:

        response = local_llm.invoke(
            prompt
        )

        return _extract_json_object(
            response.content
        )

    except Exception:

        return {
            "ambiguous": False,
            "clarification": ""
        }
        
def bm25_search(query, top_k=3):

    global bm25
    global all_chunks

    if bm25 is None:
        return []

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked = sorted(
        zip(all_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk
        for chunk, score
        in ranked[:top_k]
    ]

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(query, docs, top_k=3):
    if not docs:
        return []
    
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked[:top_k]
    ]

def answer_question(
    question,
    is_subquestion=False
):
    global retriever
    global chat_history
    global conversation_summary

    try:
        # =====================
        # PREPROCESS
        # =====================
        question = normalize_query(question)
        question = strip_prompt_injection(question)
        question = expand_acronyms(question)

        # =====================
        # ROUTING
        # =====================
        question_source = classify_question_source(
            question
        )

        print(
            "QUESTION SOURCE:",
            question_source
        )

        if (
            ENABLE_GENERAL_ROUTING
            and question_source == "GENERAL"
            and not is_subquestion
        ):
            answer = answer_general_question(
                question
            )

            chat_history.append(
                (question, answer)
            )

            return answer, []

        # =====================
        # DOCUMENT CHECK
        # =====================
        if retriever is None:
            return (
                "Please upload documents first.",
                []
            )

        # =====================
        # QUERY REWRITE
        # =====================
        rewritten_query = question

        if ENABLE_QUERY_REWRITE:
            rewritten_query = rewrite_query(
                question
            )

        # =====================
        # CLASSIFICATION
        # =====================
        query_type = classify_query(
            question
        )

        intent = classify_intent(
            question
        )

        print(
            "QUERY TYPE:",
            query_type
        )

        print(
            "INTENT:",
            intent
        )

        # =====================
        # DYNAMIC K
        # =====================
        k = 5

        if query_type == "holistic":
            k = 10
        elif query_type == "spanning":
            k = 8
        elif query_type == "numerical":
            k = 3
        elif query_type == "definition":
            k = 3
        elif query_type == "concept":
            k = 5

        if intent == "comparison":
            k = max(k, 10)

        elif intent == "summarization":
            k = max(k, 12)

        elif intent == "analysis":
            k = max(k, 10)

        elif intent == "troubleshooting":
            k = max(k, 8)

        elif intent == "fact_lookup":
            k = min(k, 3)

        elif intent == "find_doc":
            k = max(k, 5)

        elif intent == "howto":
            k = max(k, 7)

        print(
            f"RETRIEVAL K: {k}"
        )

        # =====================
        # RETRIEVAL
        # =====================
        document_scope = detect_document_scope(
            question
        )

        queries = [rewritten_query]

        if ENABLE_MULTI_QUERY:
            queries = generate_search_queries(
                rewritten_query
            )

        if not queries:
            queries = [rewritten_query]

        all_docs = []

        for query in queries:

            retriever.search_kwargs["k"] = k

            vector_docs = retriever.invoke(
                query
            )

            bm25_k = max(
                3,
                k // 2
            )

            bm25_docs = bm25_search(
                query,
                top_k=bm25_k
            )

            all_docs.extend(
                vector_docs
            )

            all_docs.extend(
                bm25_docs
            )

        # =====================
        # DEDUPLICATE
        # =====================
        combined_docs = []
        seen = set()

        for doc in all_docs:

            text = doc.page_content

            if text not in seen:
                seen.add(text)
                combined_docs.append(doc)

        # =====================
        # RERANK
        # =====================
        if ENABLE_RERANKING:

            combined_docs = rerank_documents(
                question,
                combined_docs,
                top_k=5
            )

        # =====================
        # DOCUMENT FILTER
        # =====================
        if document_scope:

            filtered_docs = []

            for doc in combined_docs:

                source = (
                    doc.metadata
                    .get(
                        "document_name",
                        ""
                    )
                    .lower()
                )

                if document_scope in source:
                    filtered_docs.append(
                        doc
                    )

            if filtered_docs:
                combined_docs = (
                    filtered_docs
                )

        # =====================
        # PARENT RETRIEVAL
        # =====================
        if ENABLE_PARENT_RETRIEVAL:

            combined_docs = (
                expand_to_parent_docs(
                    combined_docs
                )
            )

        # =====================
        # CONTEXT
        # =====================
        if ENABLE_CONTEXT_COMPRESSION:

            context = compress_context(
                question,
                combined_docs
            )

        else:

            context = "\n\n".join(
                doc.page_content
                for doc in combined_docs
            )

        # =====================
        # HISTORY
        # =====================
        history_text = ""

        for q, a in chat_history[-3:]:

            history_text += (
                f"\nUser: {q}"
                f"\nAssistant: {a}\n"
            )

        # =====================
        # HOLISTIC BRANCH
        # =====================
        if (
            query_type == "holistic"
            and not is_subquestion
            and len(question.split()) > 8
        ):

            print(
                "ENTERED HOLISTIC BRANCH"
            )

            sub_questions = (
                decompose_question(
                    question
                )
            )

            sub_answers = []
            sub_docs_all = []

            for sub_q in sub_questions:

                sub_answer, sub_docs = (
                    answer_question(
                        sub_q,
                        is_subquestion=True
                    )
                )

                sub_answers.append(
                    sub_answer
                )

                sub_docs_all.extend(
                    sub_docs
                )

            answer = aggregate_answers(
                question,
                sub_answers
            )

            chat_history.append(
                (question, answer)
            )

            return (
                answer,
                sub_docs_all
            )

        # =====================
        # NORMAL BRANCH
        # =====================
        print(
            "ENTERED NORMAL BRANCH"
        )

        prompt = f"""
You are a highly accurate document
question-answering system.

Conversation Summary:
{conversation_summary}

Recent History:
{history_text}

Question Type:
{query_type}

Intent:
{intent}

Use ONLY the provided context.

Rules:

- For fact_lookup:
answer briefly and precisely.

- For comparison:
compare all relevant items.

- For summarization:
provide a concise summary.

- For troubleshooting:
explain causes and solutions.

- For analysis:
provide detailed reasoning.

- Preserve exact numerical values.
- Do not approximate numbers.

- If answer not found,
return exactly:

"Not found in document."

Context:
{context}

Question:
{question}

Answer:
"""

        start = time.time()

        answer = generate_answer(
            prompt
        )

        review = self_evaluate(
            question,
            context,
            answer
        )

        if (
            ENABLE_SELF_CORRECTION
            and review["confidence"] < 7
        ):
            answer = (
                self_correct_answer(
                    question,
                    context,
                    answer,
                    review["feedback"]
                )
            )

            review = self_evaluate(
                question,
                context,
                answer
            )

        if ENABLE_ANSWER_VERIFICATION:

            verification = (
                verify_answer(
                    question,
                    context,
                    answer
                )
            )

            print(
                "Verification:",
                verification["supported"]
            )

        llm_time = (
            time.time() - start
        )

        print(
            f"LLM Time: "
            f"{llm_time:.2f} sec"
        )

        print(
            "Self Evaluation Confidence:",
            review["confidence"]
        )

        chat_history.append(
            (question, answer)
        )

        if len(chat_history) >= 10:

            summarize_conversation()

            chat_history = (
                chat_history[-3:]
            )

        return (
            answer,
            combined_docs
        )

    except Exception as e:

        print(
            "\nANSWER QUESTION ERROR:"
        )

        print(str(e))

        traceback.print_exc()

        return (
            "Answer Generation Failed",
            []
        )

def generate_search_queries(question):
    print("MULTI QUERY CALLED")
    prompt = f"""
    Generate 3 different search queries
    for retrieving information.

    Question:
    {question}

    Return one query per line.
    """

    response = local_llm.invoke(prompt)
    text = response.content
    queries = [
        q.strip()
        for q in text.split("\n")
        if q.strip()
    ]

    return queries[:3]

def decompose_question(question):

    prompt = f"""
Break this question into
independent sub-questions.

Question:
{question}

Return one per line.
"""

    response = local_llm.invoke(prompt)

    return [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

def load_test_set(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(
        f"Loaded {len(test_questions)} test questions"
    )

    return test_questions

def detect_document_scope(question):

    question = question.lower()

    if (
        "niladri" in question
        or "leetcode" in question
        or "cgpa" in question
        or "resume" in question
    ):
        return "resume"

    return None

def evaluate_retrieval(item, retrieved_docs):

    keywords = item["keywords"]

    retrieved_text = " ".join(
        doc.page_content.lower()
        for doc in retrieved_docs
    )

    matched = 0

    for keyword in keywords:

        if keyword.lower() in retrieved_text:
            matched += 1

    coverage = matched / len(keywords)

    reciprocal_rank = 0

    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in keywords
        ):

            reciprocal_rank = 1 / rank

            break

    recall = matched > 0

    return {
        "coverage": coverage,
        "mrr": reciprocal_rank,
        "recall": recall
    }

def _extract_json_object(text: str) -> dict:
    """
    Tries to parse strict JSON first.
    If the model adds extra text, extracts the first JSON object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))

def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str
) -> dict:

    prompt = f"""
    You are a strict evaluation judge.

    Question:
    {question}

    Reference Answer:
    {reference_answer}

    Generated Answer:
    {generated_answer}

    Scoring Rules:

    10 = Perfect

    9 = Excellent

    8 = Very Good

    7 = Good

    6 = Acceptable

    5 = Partially Correct

    4 = Weak

    3 = Poor

    2 = Very Poor

    1 = Completely Incorrect

    0 = No Answer

    Return ONLY JSON:

    {{
        "accuracy": 0-10,
        "completeness": 0-10,
        "relevance": 0-10,
        "feedback": "short explanation"
    }}
    """
    

    response = judge_llm.invoke(prompt)

    content = response.content.strip()

    result = _extract_json_object(content)

    print("\nRAW JUDGE JSON")
    print(result)

    return result

def evaluate():
    ENABLE_HUMAN_CLARIFICATION = False
    global conversation_summary

    test_questions = load_test_set(
        "test.json"
    )

    conversation_summary = ""
    chat_history.clear()

    correct = 0
    total_coverage = 0
    total_mrr = 0
    total_recall = 0

    total_judge_accuracy = 0
    total_judge_completeness = 0
    total_judge_relevance = 0

    failed_questions = []

    category_stats = {}
    detailed_results = []
    for item in test_questions:

        generated_answer, combined_docs = answer_question(
            item["question"]
        )
        print("\nAFTER answer_question()")
        print(repr(generated_answer))

        chat_history.clear()
        
        try:
            print("\nBEFORE JUDGE")
            print(repr(generated_answer))

            judge = judge_answer(
                item["question"],
                item["reference_answer"],
                generated_answer
            )

        except Exception as e:

            print("Judge Error:", e)

            judge = {
                "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "feedback": "Judge failed"
                }

        total_judge_accuracy += judge["accuracy"]
        total_judge_completeness += judge["completeness"]
        total_judge_relevance += judge["relevance"]

        print("\n-------------------")
        print("\nBEFORE FINAL REPORT")
        print(repr(generated_answer))

        print("Question:", item["question"])
        print("Expected:", item["reference_answer"])
        print("Actual:", generated_answer)

        print("\nJUDGE SCORES")
        print("Accuracy:", judge["accuracy"])
        print("Completeness:", judge["completeness"])
        print("Relevance:", judge["relevance"])
        print("Feedback:", judge["feedback"])

        metrics = evaluate_retrieval(
            item,
            combined_docs
        )
        
        total_coverage += metrics["coverage"]
        total_mrr += metrics["mrr"]
        total_recall += metrics["recall"]
        
        print(
            f"Coverage={metrics['coverage']:.2f}"
            )

        print(
            f"MRR={metrics['mrr']:.2f}"
            )

        print(
            f"Recall={metrics['recall']}"
            )

        keywords = item["keywords"]
        category = item["category"]

        if category not in category_stats:

            category_stats[category] = {
                "correct": 0,
                "total": 0
            }

        category_stats[category]["total"] += 1

        detailed_results.append(
            {
                "question": item["question"],

                "category": category,

                "expected": item["reference_answer"],

                "actual": generated_answer,

                "judge_accuracy": judge["accuracy"],

                "judge_completeness":
                judge["completeness"],

                "judge_relevance": judge["relevance"],

                "judge_feedback": judge["feedback"],

                "retrieved_chunks": [
                    doc.page_content[:500]
                    for doc in combined_docs
                ]
            }
        )

        
        semantic_result = judge

        score = (
            (
                semantic_result["accuracy"]
                +
                semantic_result["completeness"]
                +
                semantic_result["relevance"]
            )
            / 30
        )

        if score >= 0.7:

            correct += 1

            category_stats[category]["correct"] += 1

        else:

            failed_questions.append(
                {
                    "question": item["question"],
                    "expected": item["reference_answer"],
                    "actual": generated_answer,
                    "category": category
                }
            )

    accuracy = (
        correct /
        len(test_questions)
    ) * 100

    print(
        f"\nAccuracy: {accuracy:.2f}%"
    )

    print("\n========================")
    print("CATEGORY PERFORMANCE")
    print("========================")

    for category, stats in category_stats.items():

        category_accuracy = (
            stats["correct"] /
            stats["total"]
        ) * 100

        print(
            f"{category}: "
            f"{category_accuracy:.2f}% "
            f"({stats['correct']}/{stats['total']})"
        )


    avg_coverage = (total_coverage /len(test_questions))

    avg_mrr = (total_mrr /len(test_questions))

    recall_at_k = (total_recall /len(test_questions))
    
    print("\n========================")
    print("RETRIEVAL PERFORMANCE")
    print("========================")

    print(
        f"Keyword Coverage: "
        f"{avg_coverage:.2%}"
    )

    print(
        f"Recall@K: "
        f"{recall_at_k:.2%}"
    )

    print(
        f"MRR: "
        f"{avg_mrr:.4f}"
    )

    print("\n========================")
    print("FAILED QUESTIONS")
    print("========================")

    for item in failed_questions:

        print("\nQuestion:")
        print(item["question"])

        print("\nCategory:")
        print(item["category"])

        print("\nExpected:")
        print(item["expected"])

        print("\nActual:")
        print(item["actual"])

        print("\n-------------------")

    n = len(test_questions)

    print("\n========================")
    print("JUDGE PERFORMANCE")
    print("========================")

    print(f"Average Accuracy Score: {total_judge_accuracy / n:.2f}/10")
    print(f"Average Completeness Score: {total_judge_completeness / n:.2f}/10")
    print(f"Average Relevance Score: {total_judge_relevance / n:.2f}/10")


    results = {
        "accuracy": accuracy,
        "coverage": avg_coverage * 100,
        "recall": recall_at_k * 100,
        "mrr": avg_mrr,

        "judge_accuracy": total_judge_accuracy / n,
        "judge_completeness": total_judge_completeness / n,
        "judge_relevance": total_judge_relevance / n,

        "category_stats": category_stats,

        "failed_questions": failed_questions,

        "detailed_results":detailed_results,
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

def run_evaluation():

    evaluate()

    return "Evaluation Complete. Open dashboard.py and click Load Evaluation."

def compress_context(question,docs):

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
    You are a retrieval compressor.

    Question:
    {question}

    Context:
    {context}

    Extract ONLY the information
    relevant to answering the question.

    Return concise notes.
    """

    response = judge_llm.invoke(prompt)
    return response.content

def rewrite_query(query):

    history_text = ""

    for q, a in chat_history[-3:]:

        history_text += (
            f"\nUser:{q}\n"
            f"Assistant:{a}\n"
        )

    prompt = f"""
    Conversation Summary:
    {conversation_summary}

    Recent History:
    {history_text}
    
    Current Question:
    {query}

    Rewrite the question
    as a standalone search query.

    Return only query.
    """

    response = (
        local_llm.invoke(
            prompt
        )
    )

    return (
        response.content
        .strip()
    )

def strip_prompt_injection(
    query
):

    blocked = [

        "ignore previous instructions",

        "forget all instructions",

        "system prompt",

        "reveal prompt",

        "developer instructions",

        "act as"

    ]

    cleaned = query

    for item in blocked:

        cleaned = cleaned.replace(
            item,
            ""
        )

    return cleaned

ACRONYMS = {

    "ml": "machine learning",

    "ai": "artificial intelligence",

    "llm": "large language model",

    "rag": "retrieval augmented generation",

    "api": "application programming interface",

    "db": "database",

    "nlp": "natural language processing"
}

def expand_acronyms(query):

    words = query.split()

    expanded = []

    for word in words:

        expanded.append(
            ACRONYMS.get(
                word.lower(),
                word
            )
        )

    return " ".join(
        expanded
    )

def normalize_query(query):

    query = expand_acronyms(query)
    
    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    query = re.sub(
        r"[\x00-\x1F\x7F]",
        "",
        query
    )

    return query

def aggregate_answers(
    question,
    answers
):

    prompt = f"""
Question:
{question}

Sub Answers:
{answers}

Combine into one answer.
"""

    response = local_llm.invoke(prompt)

    return response.content

def summarize_conversation():

    global chat_history
    global conversation_summary

    history_text = ""

    for q, a in chat_history:

        history_text += (
            f"\nUser:{q}\n"
            f"Assistant:{a}\n"
        )

    prompt = f"""
Summarize the important information
from this conversation.

Conversation:
{history_text}

Return concise summary.
"""

    response = local_llm.invoke(prompt)

    conversation_summary = (
        response.content
    )

def expand_to_parent_docs(
    docs
):

    expanded = []

    for doc in docs:

        parent_id = doc.metadata.get(
            "parent_id"
        )

        parent_text = parent_docs.get(
            parent_id
        )

        if parent_text:

            doc.page_content = parent_text

        expanded.append(doc)

    return expanded

with gr.Blocks() as demo:

    history = gr.State([])

    document = gr.File(
        file_count="multiple",
        file_types=[".pdf", ".docx"]
    )

    status = gr.Textbox(
        label="Knowledge Base Status"
    )

    document.change(
        fn=process_document,
        inputs=document,
        outputs=status
    )

    chatbot = gr.Chatbot(
        height=600
    )

    msg = gr.Textbox(
        placeholder="Ask a question..."
    )

    msg.submit(
        fn=chat,
        inputs=[msg, history],
        outputs=[chatbot, history, msg]
    )
    clear_btn = gr.Button("Clear Chat")
    clear_btn.click(
        lambda: ([], []),
        outputs=[chatbot, history]
    )
    eval_btn = gr.Button(
        "Run Evaluation"
    )

    eval_status = gr.Textbox(
        label="Evaluation Status"
    )
    
    eval_btn.click(
        fn=run_evaluation,
        outputs=eval_status
    )
if __name__ == "__main__":
    demo.launch()