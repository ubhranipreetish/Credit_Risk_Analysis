# Project 10: Intelligent Credit Risk Scoring & Agentic Lending Decision Support

## From Financial Risk Modeling to Automated Lending Advice

### Project Overview
This project involves the design and implementation of an **AI-driven credit analytics system** that evaluates borrower profiles and predicts credit risk, and evolves into an agentic AI lending decision assistant.

- **Milestone 1:** Classical machine learning techniques applied to historical borrower financial data to predict creditworthiness, default probability, and loan risk.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about borrower risk, retrieves regulatory guidelines (RAG), and generates structured lending recommendations.

---

### Constraints & Requirements
- **Team Size:** 3–4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Credit Risk Scoring (Mid-Sem)
**Objective:** Design and implement a machine learning-based credit risk scoring system using historical borrower data, focusing strictly on classical ML pipelines *without LLMs*.

**Key Deliverables:**
- Problem understanding & Lending use-case context.
- System architecture diagram.
- Working local application with UI (Streamlit/Gradio).
- Model performance evaluation report (Accuracy, ROC-AUC, Confusion Matrix).

#### Milestone 2: Agentic AI Lending Decision Assistant (End-Sem)
**Objective:** Extend the scoring system into an agentic AI assistant that reasons about borrower risk, retrieves financial regulations, and generates structured lending reports.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured lending assessment report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | Correct ML technique application, Quality of Preprocessing & Feature Engineering, UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Quality of Agentic Reasoning, Correct RAG Integration, Output Clarity, Deployment Success. |
