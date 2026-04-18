# AI-Based Credit Risk Scoring and Agentic Lending Decision System

## 1. End-to-End System Overview

This project is designed as a two-layer intelligent lending system:

- **Milestone 1 (ML System):** predicts borrower default risk using classical machine learning.
- **Milestone 2 (Agentic AI System):** builds on ML outputs to generate explainable, regulation-aware lending recommendations.

The complete pipeline is:

1. Borrower data is collected from the UI.
2. Inputs are validated and transformed into model-ready features.
3. ML model predicts risk class and default probability.
4. Agentic workflow consumes ML output and feature-level risk drivers.
5. RAG retrieves relevant regulatory context from vector index.
6. LLM reasons over profile + risk + regulations.
7. Final structured lending report is generated.

So the system evolves from **prediction** (Milestone 1) to **reasoned decision support** (Milestone 2).

---

## 2. Milestone 1: Machine Learning Credit Risk System

### Input Data
Borrower profile includes core lending attributes such as:

- Age
- Annual income
- Employment length
- Loan amount
- Interest rate
- Loan-to-income ratio (`loan_percent_income`)
- Credit history length
- Home ownership
- Loan intent
- Loan grade
- Historical default flag

### Preprocessing
The system performs:

- Schema and value validation (types, ranges, allowed categories)
- Normalization of categorical input values
- Use of trained sklearn pipelines that include preprocessing + model in one artifact

### Models Used
Two classical models are part of Milestone 1:

- **Logistic Regression**
- **Decision Tree**

(Repo and documentation also align with tree-based methods in the evaluation narrative.)

### Output
Milestone 1 provides:

- Risk class (High Risk / Low Risk)
- Default probability
- Model used

### Evaluation
The training/evaluation pipeline reports:

- **Accuracy**
- **ROC-AUC**
- **Confusion Matrix**

Metrics are stored in `reports/metrics.json` and surfaced in the UI.

### UI Role
Milestone 1 now has a dedicated tab in Streamlit for:

- Borrower input
- Model selection (Logistic / Decision Tree)
- ML risk result display
- Evaluation snapshot display

---

## 3. Transition to Agentic AI (Core Insight)

ML-only risk scores are not sufficient for real lending decisions because:

- Scores alone do not provide auditable narrative reasoning.
- Borderline cases require contextual and policy-aware analysis.
- Financial decisioning demands traceable references and compliance awareness.

Therefore, Milestone 2 converts ML output into structured agent input:

- ML probability and label become primary risk signal.
- Feature explanations become supporting evidence.
- Regulation retrieval provides external grounding.

This shift satisfies rubric expectations around reasoning quality, explainability, and practical decision support.

---

## 4. Milestone 2: Agentic Lending Decision System

Milestone 2 is orchestrated as a **stateful LangGraph workflow**.

### Inputs

- Borrower profile
- ML prediction output (risk label, probability)
- Feature importance/explanation context

### Components

- **LLM Reasoner** for narrative decision generation
- **RAG subsystem** with FAISS index for regulatory retrieval
- **LangGraph** for node-based state transitions and routing

### Workflow Steps

1. **Profile Understanding / Validation**
   - Validate borrower schema and produce canonical feature row.

2. **Risk Analysis**
   - Run ML inference and classify risk tier (clear low / borderline / clear high).

3. **Regulatory Retrieval**
   - Build contextual query from ML risk and top features.
   - Retrieve relevant guidelines from vector store.

4. **Decision Generation**
   - LLM reasons using ML output + explanations + retrieved context.
   - Confidence scoring applied.
   - Reflection loop can trigger for lower-confidence decisions.

5. **Structured Report Creation**
   - Output normalized to strict JSON schema for consistent downstream rendering.

---

## 5. RAG (Retrieval-Augmented Generation)

### What is Retrieved
The system retrieves semantically similar chunks from indexed financial/regulatory documents.

### Why It Is Necessary
RAG reduces hallucinations by grounding decisions in retrieved evidence instead of free-form model memory.

### Integration into Agent Pipeline
RAG is called after query construction and before final decision generation, so the LLM receives contextual evidence while composing recommendation and references.

---

## 6. Structured Output Format

Final output is intentionally schema-bound and includes:

1. **Borrower Profile Summary**
2. **Risk Analysis**
3. **Lending Decision** (`APPROVE` / `REJECT` / `CONDITIONAL`)
4. **Regulatory References**
5. **Disclaimer**

Benefits:

- Consistent UI rendering
- Easier auditing and compliance review
- Safer integration with APIs and reports

---

## 7. System Architecture (Big Picture)

### Frontend
- **Streamlit UI** with tabbed navigation:
  - Milestone 1 tab: ML scoring and metrics view
  - Milestone 2 tab: Agentic decision support

### Backend
- Service layer validates request and executes workflow.
- Agent nodes handle prediction, explanation, retrieval, reasoning, and reflection.

### Vector DB / Index
- FAISS index stores document embeddings and chunk metadata.

### LLM
- Used for structured reasoning and narrative decision synthesis.

### Data Flow
UI -> Schema Validation -> ML Prediction -> Agent Routing -> RAG Retrieval -> LLM Reasoning -> Structured Decision -> UI/API Response

---

## 8. Mapping to Evaluation Rubric

### Technical Implementation
- Agentic orchestration (LangGraph)
- RAG integration (FAISS retrieval)
- Multi-step reasoning with confidence/reflective behavior
- Structured output quality

### Code Quality
- Modular repo structure (agent, backend, schemas, services, src)
- Separation of concerns across ML, API, and agent logic

### Deployment
- Streamlit app architecture supports hosted deployment requirement (Streamlit Cloud/Render/HF Spaces)

### Report
- Pipeline is documentable end-to-end for LaTeX technical report
- Evaluation metrics and workflow states are explicit

### Video
- Demonstration can show:
  1. Milestone 1 prediction flow
  2. Milestone 2 agent + RAG + structured report flow

---

## 9. Key Design Decisions

### Why LangGraph
Needed for stateful, conditional, multi-node execution with routing and iterative reflection.

### Why RAG
Required to ground decisions in external regulations and reduce hallucinated policy statements.

### Why Structured Outputs
Critical for auditability, reliability, and deterministic UI/API handling.

### How Hallucination is Controlled
- Retrieval grounding before generation
- Canonical schema normalization
- Reference validation behavior
- Fallback and warning signals when context is missing

---

## 10. Real-World Significance

This system mirrors practical banking decision support:

- ML-based risk screening
- Analyst-style reasoning layer
- Policy/regulatory grounding
- Human-review disclaimer

It demonstrates responsible AI principles in financial risk assessment:

- Explainability
- Traceability
- Conservative decision behavior
- Human-in-the-loop final authority

---

## Short Viva Pitch (Optional)

"Milestone 1 predicts borrower default risk using classical ML with measurable performance metrics. Milestone 2 upgrades that prediction engine into an agentic lending assistant that reasons over model outputs, retrieves regulatory evidence through RAG, and produces a structured, auditable recommendation for human decision-makers."