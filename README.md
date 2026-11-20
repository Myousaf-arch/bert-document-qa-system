<div align="center">

<h1>BERT DOCUMENT QA SYSTEM</h1>
<p><strong>SEMANTIC VECTOR SEARCH &amp; CONTEXTUAL QUESTION ANSWERING ENGINE FOR PDF DOCUMENTS</strong></p>

<p>
  <img src="https://img.shields.io/badge/PYTHON-3.12-000000?style=for-the-badge&logo=python&logoColor=00F2FE" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/TRANSFORMERS-BERT--SQUAD-000000?style=for-the-badge&logo=huggingface&logoColor=00F2FE" alt="BERT SQuAD" />
  <img src="https://img.shields.io/badge/VECTOR_SEARCH-SEMANTIC_INDEX-000000?style=for-the-badge&logo=python&logoColor=00F2FE" alt="Vector Search" />
  <img src="https://img.shields.io/badge/INTERFACE-STREAMLIT-000000?style=for-the-badge&logo=streamlit&logoColor=00F2FE" alt="Streamlit" />
</p>

</div>

---

## THE PROBLEM IT SOLVES

Modern work and study drown people in long PDFs—research papers, policy manuals, contracts, technical guides, and multi-chapter notes. Traditional keyword search (`Ctrl + F`) fails whenever a document uses different wording than your question, forcing you to skim dozens of pages manually and waste hours hunting for a single concrete answer.

**BERT Document QA System** solves this by turning static PDF files into searchable semantic context and extracting direct, span-level answers to natural language questions in seconds.

---

## WHY IT IS BETTER THAN MANUAL SEARCH

* **MEANING OVER EXACT KEYWORDS**
  Finds relevant passages based on semantic intent rather than matching identical words, catching concepts even when phrased differently across pages.

* **PINPOINT ANSWERS INSTEAD OF WALLS OF TEXT**
  Combines semantic chunk retrieval (`vector_store.py`) with a SQuAD-tuned **BERT** reader so you get targeted answers grounded in the uploaded document instead of guessing through paragraphs.

* **MODULAR & CLEAN ARCHITECTURE**
  Separates PDF extraction and text chunking (`utils.py`), vector indexing and retrieval (`vector_store.py`), and the interactive web interface (`app.py`) for easy maintenance and scaling.

* **FAST INTERACTIVE WORKFLOW**
  Built with **Streamlit** so you can upload a document, index its content, and ask follow-up questions in one clean browser workspace.

---

## HOW IT MAKES LIFE EASIER

* **SAVES HOURS OF READING & SCANNING**
  Jump straight to exact facts, definitions, clauses, or figures inside dense PDFs without reading every page line by line.

* **REDUCES COGNITIVE OVERLOAD**
  Keeps your focus on decisions, learning, and execution rather than manual page-flipping and copy-pasting notes.

* **INSTANT CLARITY ON COMPLEX MATERIAL**
  Lets you test specific questions against unfamiliar technical or academic documents immediately and verify the exact source context.

---

## WHERE YOU CAN USE IT

* **ACADEMIC RESEARCH & STUDY PREP**
  Query textbooks, lecture PDFs, assignments, and research papers while preparing exams or literature reviews.

* **BUSINESS & CLIENT OPERATIONS**
  Search proposals, service agreements, SOPs, and product documentation to answer client or team questions fast.

* **TECHNICAL & PRODUCT DOCUMENTATION**
  Look up specifications, architecture notes, and compliance guidelines across multi-page engineering PDFs.

* **PERSONAL KNOWLEDGE & ARCHIVES**
  Turn saved guides, reports, and reference PDFs into an interactive personal Q&A assistant.

---

## SYSTEM CAPABILITIES

| MODULE | RESPONSIBILITY |
| :--- | :--- |
| **Document Pipeline (`utils.py`)** | Extracts text from uploaded PDFs and splits content into clean overlapping chunks |
| **Retrieval Engine (`vector_store.py`)** | Generates embeddings and retrieves the most relevant passages for each query |
| **QA Reader Model** | Fine-tuned **BERT** extractive question-answering model for span extraction |
| **Web Application (`app.py`)** | Interactive **Streamlit** interface for document upload, search, and Q&A |

---

## QUICK START

```bash
git clone [https://github.com/Myousaf-arch/bert-document-qa-system.git](https://github.com/Myousaf-arch/bert-document-qa-system.git)
cd bert-document-qa-system
pip install -r requirements.txt
streamlit run app.py
