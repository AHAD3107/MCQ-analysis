# KnowledgeTrace MCQ: Advanced Machine Learning Analytics & Cognitive Diagnostic Engine

## Academic Session 2026-2027 | CSE Mega Project
**Branch:** `ml-advanced-analytics`  
**System Architecture:** Neuro-Symbolic Cognitive Diagnosis, Psychometric Item Profiling, and Agentic Remediation  

---

## 1. Executive Overview

Standard Learning Management Systems (LMS) and computer-based testing engines evaluate student performance using raw percentage arithmetic (for example, 7/10 = 70%). This scalar score is pedagogically uninformative because it fails to address five fundamental realities of human testing:

1. **Guessing Blindness:** In a standard 4-option multiple-choice question, random guessing carries a 25% baseline success probability. Traditional scoring awards full credit for lucky guesses.
2. **Careless Slip Penalties:** A high-mastery student who understands a topic may make an accidental typographical or reading error. Traditional grading treats this error identically to complete ignorance.
3. **Multi-Skill Confounding:** Engineering and technical assessments test compound skills. If an assessment item requires both SQL Execution and Query Optimization, a single score cannot determine which sub-skill failed.
4. **Compounding Prerequisite Debt:** When foundational concepts are unmastered, failures compound downstream. Traditional systems fail to trace learning deficits back to their root blockers.
5. **Item Psychometric Defects:** Exam questions often contain ambiguous phrasing or misleading trick distractors that confuse high-performing students. Traditional systems assume every exam question is flawless.

The `ml-advanced-analytics` branch implements an interpretable, non-blocking, publication-grade **Neuro-Symbolic Cognitive Diagnostic Architecture**. It integrates **Bayesian Knowledge Tracing (BKT)**, the **Multi-Attribute DINA Model**, **Response-Time Effort (RTE)** behavioral telemetry, **Multivariate Logistic Regression** slip detection, **Point-Biserial Item Discrimination** ($r_{pb}$), and **NetworkX Directed Acyclic Graph (DAG)** causal analysis into a unified, responsive platform.

---

## 2. End-to-End System Workflow

The complete system workflow operates across five distinct phases:

```
[Phase 1: Curriculum & Test Authoring]
                  │
                  ▼
[Phase 2: High-Precision Student Telemetry]
                  │
                  ▼
[Phase 3: Asynchronous Background ML Inference]
                  │
                  ▼
[Phase 4: Teacher Cohort Psychometrics & Heatmaps]
                  │
                  ▼
[Phase 5: Closed-Loop AI Remediation & Assignment]
```

### Phase 1: Curriculum & Test Authoring (Teacher)
* **Topological Prerequisite Control:** Teachers define dependency links between concepts (e.g., Recursion -> Binary Trees). The system uses NetworkX to enforce an acyclic invariant, immediately rejecting circular logic ($A \to B \to A$) with an HTTP 400 validation error.
* **Multi-Skill Q-Matrix Tagging:** Each question is mapped to one primary concept and multiple secondary skill attributes in the database `question_skills` table.
* **Two Creation Workflows:**
  1. *Bulk Question Import:* Teachers paste plain-text blocks (`Q1 - Concept: Name ...`), which the system parses, extracts options, validates answer keys, and auto-links concepts.
  2. *AI Curriculum Generator:* Teachers enter a topic name, and the agent auto-extracts concepts, validates DAG links, and synthesizes candidate multi-skill MCQs for instructor review.

### Phase 2: High-Precision Student Telemetry (Student)
* As the student reads and solves questions, a client-side performance timer captures millisecond latency (`response_time_ms`).
* **Non-Blocking Ingestion:** When the student submits an answer, FastAPI writes the record to PostgreSQL and returns an HTTP 200 payload in under 15 milliseconds. The student advances immediately with zero server lag.

### Phase 3: Asynchronous Background ML Inference (FastAPI BackgroundTasks)
A decoupled worker executes multi-model cognitive evaluation in the background:
1. **Bayesian Knowledge Tracing (BKT):** Updates latent mastery probabilities after each attempt using Bayes' rule.
2. **DINA Conjunctive Solver:** Evaluates student latent attribute vectors across the $2^K$ skill space to isolate which specific sub-skill caused an incorrect response.
3. **Cognitive Latency Profiler:** Compares actual solving latency against calibrated reading and context-switching expectations ($T_{\text{expected}}$) to assign one of four behavioral quadrants.
4. **Careless Slip Detector:** Uses multivariate logistic regression to identify when a high-mastery student suffered an accidental slip.

### Phase 4: Teacher Cohort Psychometrics (Teacher Dashboard)
When the instructor opens test results, the engine aggregates cohort performance:
* **Student x Concept Mastery Heatmap:** A visual matrix showing continuous DINA mastery percentages across all students and curriculum topics.
* **Point-Biserial Discrimination ($r_{pb}$):** Calculates Pearson correlation between individual question accuracy and total test score to tag items as STRONG, FAIR, or REVIEW NEEDED.
* **Distractor Magnet Analysis:** Aggregates option selection distributions and flags the top incorrect option with a Top Trap indicator.
* **Prerequisite Bottleneck Graph:** Evaluates class-wide mastery against the DAG to highlight root blocker nodes choking downstream comprehension.

### Phase 5: Closed-Loop AI Remediation & Assignment
* The teacher clicks "Analyze Class with AI Agent".
* The agent reviews cohort metrics and writes a concise executive briefing advising what to reteach first.
* The agent automatically drafts a targeted 3-to-5 question remedial quiz targeting the identified root bottlenecks.
* **Human-in-the-Loop Safety:** The teacher can edit stems, modify options, and adjust answer keys in a live preview.
* **Selective Assignment:** The teacher selects specific struggling student checkboxes and clicks "Approve & Assign". The remedial quiz is created and dispatched immediately to those students.

---

## 3. Mathematical Foundations & Formulations (Detailed Explanations)

This section details the mathematical and probabilistic models that power the engine, paired with plain-English intuition, variable breakdowns, and real-world interpretations.

---

### 1. DINA Multi-Skill Conjunctive Model (The AND-Gate)
*File:* `backend/app/services/dina_service.py`

#### What It Means in Plain English:
Real-world engineering questions rarely test a single, isolated skill. A technical database question might require both *SQL Query Execution* and *B-Tree Indexing*. Under the DINA (*Deterministic Inputs, Noisy "And" gate*) model, success requires **all** required skills simultaneously. If a student understands SQL syntax but has no grasp of Indexing, they cannot solve the problem through domain knowledge alone—their knowledge collapses to zero, and they are forced to guess.

#### Mathematical Formulation:
$$\eta_{ij} = \prod_{k=1}^K \alpha_{ik}^{q_{jk}}$$

#### Variable Definitions:
* $\eta_{ij} \in \{0, 1\}$: The ideal latent knowledge response of student $i$ on question $j$.
* $\alpha_{ik} \in \{0, 1\}$: The latent mastery state of student $i$ on concept $k$ ($1$ = Mastered, $0$ = Unmastered).
* $q_{jk} \in \{0, 1\}$: The binary entry in the curriculum Q-matrix ($1$ if question $j$ tests concept $k$, $0$ otherwise).
* $K$: Total number of concepts in the curriculum block.

#### Concrete Numerical Example:
Suppose Question 1 requires Concept 1 and Concept 2 ($\mathbf{q}_1 = [1, 1]$).
* **Student Alice** has mastered both concepts ($\boldsymbol{\alpha}_{\text{Alice}} = [1, 1]$):
  $$\eta_{\text{Alice}, 1} = 1^1 \times 1^1 = 1 \quad (\text{Possesses full required knowledge})$$
* **Student Bob** has mastered Concept 1 but failed Concept 2 ($\boldsymbol{\alpha}_{\text{Bob}} = [1, 0]$):
  $$\eta_{\text{Bob}, 1} = 1^1 \times 0^1 = 1 \times 0 = 0 \quad (\text{Cannot solve by knowledge; must guess})$$

---

### 2. DINA Item Response Likelihood (Guessing Floor vs Mastery Ceiling)
*File:* `backend/app/services/dina_service.py`

#### What It Means in Plain English:
Having the knowledge does not guarantee a correct answer (humans make typos), and lacking the knowledge does not guarantee a wrong answer (students can guess lucky). This formula acts as a smooth statistical bridge connecting latent mental knowledge ($\eta_{ij}$) to observable real-world test results ($X_{ij} = 1$ or $0$).

#### Mathematical Formulation:
$$P(X_{ij} = 1 \mid \boldsymbol{\alpha}_i) = g_j^{1 - \eta_{ij}} \cdot (1 - s_j)^{\eta_{ij}}$$

#### Constants & Empirical Rationale:
* **Guessing Floor ($g_j = 0.25$):** On a standard 4-option multiple-choice question, a student who knows nothing has a 1-in-4 random chance ($25\%$) of choosing the correct option.
* **Careless Slip Rate ($s_j = 0.10$):** Even students with complete mastery have an estimated $10\%$ rate of typographical errors, perceptual slips, or misreading options.
* **Mastery Ceiling ($1 - s_j = 0.90$):** A fully knowledgeable student has a $90\%$ probability of answering correctly.

#### Concrete Numerical Example:
* If Student Alice has $\eta = 1$:
  $$P(\text{Correct}) = 0.25^{1 - 1} \times (1 - 0.10)^1 = 0.25^0 \times 0.90^1 = 1 \times 0.90 = 0.90 \text{ (90% chance)}$$
* If Student Bob has $\eta = 0$:
  $$P(\text{Correct}) = 0.25^{1 - 0} \times (1 - 0.10)^0 = 0.25^1 \times 0.90^0 = 0.25 \times 1 = 0.25 \text{ (25% chance)}$$

---

### 3. Epistemic Confidence via Normalized Quadratic Entropy
*File:* `backend/app/services/dina_service.py`

#### What It Means in Plain English:
How certain is the system about a student's diagnosis? If a student has a $50\%$ estimated mastery on a topic, that diagnosis is no better than a coin flip—the system is completely uncertain. But if their mastery is $98\%$ (or $2\%$), the evidence is overwhelming. This metric quantifies diagnostic stability using quadratic entropy.

#### Mathematical Formulation:
$$\text{Confidence}(P) = 1 - 4 \cdot P \cdot (1 - P)$$

#### Why the Multiplier 4 is Used:
The uncertainty term $P(1 - P)$ reaches its mathematical peak when $P = 0.50$, where $0.50 \times 0.50 = 0.25$. Multiplying by 4 scales this maximum uncertainty to exactly $4 \times 0.25 = 1.0$, producing $1 - 1.0 = 0.0$ ($0\%$ confidence).

#### Diagnostic Reliability Tiers:
* **At $P = 0.50$ (Total Uncertainty / Coin Flip):**
  $$\text{Confidence} = 1 - 4(0.50)(0.50) = 1 - 1.00 = 0.00 \text{ (0% Confidence - LOW TIER)}$$
* **At $P = 0.85$ (Strong Mastery Evidence):**
  $$\text{Confidence} = 1 - 4(0.85)(0.15) = 1 - 0.51 = 0.49 \text{ (49% Confidence - MEDIUM TIER)}$$
* **At $P = 0.98$ (Definitive Certified Mastery):**
  $$\text{Confidence} = 1 - 4(0.98)(0.02) = 1 - 0.0784 = 0.9216 \text{ (92.2% Confidence - HIGH TIER)}$$

---

### 4. Bayesian Knowledge Tracing (BKT) Forward Updating
*File:* `backend/app/services/mastery_service.py`

#### What It Means in Plain English:
Learning is dynamic; it evolves question by question. BKT models a student's understanding as a 2-state Hidden Markov Model (Unlearned vs Learned). When a student answers a question, Bayes' Rule updates our belief:
* A correct answer boosts our belief of mastery, discounted by the possibility of guessing.
* An incorrect answer lowers our belief of mastery, discounted by the possibility of a slip.
* In between attempts, active practice provides an incremental opportunity to transition from the unlearned state to the learned state.

#### Mathematical Formulations:

**Step 1: Observation Evidence Update (Bayes' Rule)**
$$\text{If Correct: } \quad P(L_t \mid o_t = 1) = \frac{P(L_t) \cdot (1 - P(S))}{P(L_t) \cdot (1 - P(S)) + (1 - P(L_t)) \cdot P(G)}$$

$$\text{If Incorrect: } \quad P(L_t \mid o_t = 0) = \frac{P(L_t) \cdot P(S)}{P(L_t) \cdot P(S) + (1 - P(L_t)) \cdot (1 - P(G))}$$

**Step 2: Latent Practice Learning Transition**
$$P(L_{t+1}) = P(L_t \mid o_t) + \left[1 - P(L_t \mid o_t)\right] \cdot P(T)$$

#### Parameters & Baseline Values:
* $P(L_0) = 0.20$ ($20\%$): Conservative cold-start prior for an uninstructed student.
* $P(T) = 0.15$ ($15\%$): Probability of learning the concept during the active solving attempt.
* $P(S) = 0.10$ ($10\%$): Careless slip probability.
* $P(G) = 0.25$ ($25\%$): Lucky guess probability on 4-option MCQs.

#### Concrete Numerical Example:
Start with baseline prior $P(L_0) = 0.20$. Student answers **Correct**:
1. Bayes update:
   $$P(L_1 \mid 1) = \frac{0.20 \times 0.90}{(0.20 \times 0.90) + (0.80 \times 0.25)} = \frac{0.18}{0.18 + 0.20} = \frac{0.18}{0.38} \approx 0.4737$$
2. Learning transition:
   $$P(L_2) = 0.4737 + (1 - 0.4737) \times 0.15 = 0.4737 + 0.0789 = 0.5526 \text{ (55.3% mastery)}$$
Mastery probability jumps from $20.0\%$ to $55.3\%$ in a single question.

---

### 5. Response-Time Effort (RTE) & The 4 Behavioral Quadrants
*File:* `backend/app/services/cognitive_analytics_service.py`

#### What It Means in Plain English:
Traditional quizzes treat a 2-second answer the same as a 30-second answer. Our engine separates how fast someone answered from what they knew. It computes how many seconds an average student should take based on reading length and skill complexity, then classifies the student into one of four behavioral quadrants.

#### Mathematical Formulations:
$$T_{\text{expected}} = \max\left(7000, \, 8000 + 300 \cdot W + 5000 \cdot K_j\right) \text{ ms}$$

$$\tau \text{ (Speed Ratio)} = \frac{T_{\text{actual}}}{T_{\text{expected}}}$$

#### Component Breakdown:
* **$8000\text{ ms}$ (8 seconds):** Base sentence comprehension overhead for university-level stems.
* **$300\text{ ms/word} \times W$:** Average adult technical reading speed is 200 words per minute ($\approx 300\text{ ms/word}$).
* **$5000\text{ ms/skill} \times K_j$:** Cognitive context-switching overhead per additional skill attribute in the Q-matrix.
* **$7000\text{ ms}$ (7 seconds):** Universal minimum floor time.

#### The 4 Behavioral Quadrants:
1. **Rapid Guessing ($T_{\text{actual}} < 4000\text{ ms}$):** Submissions under 4 seconds represent blind clicking without reading. It is physiologically impossible for human eyes to scan a 20-word stem and 4 options in under 4 seconds.
2. **Cognitive Fluency ($\tau \le 1.8$ and Correct):** The student solved the problem with high speed and complete accuracy. Represents automated, intuitive mastery.
3. **Deliberate Mastery ($\tau > 1.8$ and Correct):** The student took considerable time working through the problem carefully on scratch paper and arrived at the correct answer. High-effort success.
4. **Cognitive Struggle ($T_{\text{actual}} \ge 4000\text{ ms}$ and Incorrect):** The student spent adequate time thinking, but still missed the question. This proves a genuine conceptual gap rather than disengagement.

---

### 6. Point-Biserial Item Discrimination Index ($r_{pb}$)
*File:* `backend/app/routers/teacher.py`

#### What It Means in Plain English:
Instead of only testing the students, this formula **tests the exam questions themselves**. A good question should be answered correctly by the top students in the class and missed by the struggling students. If a question is tricky, ambiguous, or has an incorrect answer key, top students will get it wrong while random guessers get it right. Point-Biserial correlation mathematically detects these defective questions.

#### Mathematical Formulation:
$$r_{pb} = \frac{\sum_{i=1}^n (X_{ij} - \bar{X}_j)(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_{ij} - \bar{X}_j)^2 \cdot \sum_{i=1}^n (Y_i - \bar{Y})^2}}$$

#### Variable Definitions:
* $X_{ij} \in \{0, 1\}$: Student $i$'s score on Question $j$ ($1$ = Correct, $0$ = Incorrect).
* $\bar{X}_j \in [0.0, 1.0]$: Question pass rate (item difficulty).
* $Y_i \in [0.0, 1.0]$: Student $i$'s overall percentage on the entire exam.
* $\bar{Y} \in [0.0, 1.0]$: Cohort mean percentage across the exam.

#### Psychometric Quality Tiers:
* **$r_{pb} \ge +0.25$ (STRONG Item):** Strong positive correlation. High-scoring students got it right; low-scoring students missed it. An exemplary benchmark question.
* **$0.10 \le r_{pb} < +0.25$ (FAIR Item):** Moderate discrimination. Acceptable, but distractors could be sharpened.
* **$r_{pb} < +0.10$ or Negative (REVIEW NEEDED):** Defective question anomaly. If $r_{pb}$ is negative, the highest-performing students on the exam missed this question more frequently than failing students. This alerts the teacher to ambiguous phrasing, double negatives, or a wrong answer key.

---

### 7. Multivariate Logistic Regression Slip & Performance Model
*File:* `backend/app/services/prediction_service.py`

#### What It Means in Plain English:
Combines 5 behavioral and historical signals to predict whether a student should succeed on a given question. If the model predicts an $85\%$ success probability and the student spent 20 seconds, but submitted an incorrect choice, the system flags the error as a **Careless Slip** rather than assuming the student lacks the concept.

#### Mathematical Formulation:
$$P(X_{ij} = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}$$

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5$$

#### Weights & Empirical Rationale:
* $\beta_0 = -0.40$: Base difficulty intercept (sets uncalibrated prior log-odds $\sigma(-0.40) \approx 0.40$).
* $\beta_1 = +0.85$: Student cumulative historical accuracy across all past tests.
* $\beta_2 = +1.80$: DINA concept mastery on the question's skills (dominant positive weight).
* $\beta_3 = +1.25$: Question empirical ease / pass rate across the cohort.
* $\beta_4 = -0.45$: Skill complexity penalty per additional concept in the Q-matrix.
* $\beta_5 = +0.35$: Deliberate solving time bonus ($\tau$).
* **Careless Slip Threshold:** Flagged when $P(\text{Success}) \ge 0.78$ and $T_{\text{actual}} \ge 4000\text{ ms}$, but the student answered incorrectly.

---

### 8. Prerequisite Debt & Cohort Root Blocker Criterion
*Files:* `backend/app/services/cognitive_analytics_service.py`, `backend/app/routers/teacher.py`

Let $W = \{c \in V \mid \bar{\alpha}_c < 0.50\}$ denote the set of unmastered concepts.

#### Prerequisite Debt (The Domino Effect for Individual Students):
$$\text{Prerequisite Debt} = \frac{\big|\{c \in W \mid \text{predecessors}(c) \cap W \neq \emptyset\}\big|}{|W|}$$

* **Plain English:** If a house's foundation has cracks, the second floor will collapse. When a student fails an advanced topic, is it because the advanced topic is hard, or because their foundational prerequisite was already broken? Prerequisite Debt measures the fraction of failed topics that were directly caused by unmastered parent topics in the DAG. High debt warns the student to stop studying advanced topics and remediate the basics first.

#### Cohort Root Blocker (The Highway Bottleneck for Teachers):
$$\text{RootBlocker}(c) \iff \left(\bar{\alpha}_c < 0.55\right) \land \left(\exists s \in \text{successors}(c) : \bar{\alpha}_s < 0.65\right)$$

* **Plain English:** Imagine a traffic jam at the first toll booth on a highway; towns down the road will be empty not because their roads are bad, but because the root toll booth is blocked. In a classroom, if the class average mastery on a root concept $c$ is below $55\%$ ($\bar{\alpha}_c < 0.55$) and at least one downstream dependent topic also fails ($\bar{\alpha}_s < 0.65$), concept $c$ is flagged on the teacher graph with a **pulsing red halo**. This instructs the teacher: *"Do not waste time reteaching the advanced topics; spend 15 minutes fixing the root blocker, and the rest of the syllabus will unlock."*

---

---

## 4. Repository & File Architecture Map

```
MCQ-analysis/
├── backend/
│   ├── app/
│   │   ├── models/                    # Database ORM Schemas (SQLAlchemy)
│   │   │   ├── user.py                # User, Roles (Teacher, Student)
│   │   │   ├── concept.py             # Concept, ConceptEdge (DAG Links)
│   │   │   ├── test.py                # Test, Question, Option, QuestionSkill (Q-Matrix)
│   │   │   ├── response.py            # Response (is_correct, response_time_ms)
│   │   │   └── diagnosis.py           # StudentConceptMastery, CognitiveDiagnosisProfile
│   │   ├── schemas/                   # Pydantic Input/Output Schemas
│   │   │   ├── concept.py             # ConceptRead, ConceptEdgeCreate
│   │   │   ├── question.py            # QuestionCreate (skill_ids), OptionCreate
│   │   │   └── agent.py               # CurriculumDraft, ClassIntervention schemas
│   │   ├── routers/                   # API Endpoints
│   │   │   ├── student.py             # POST /student/responses (Sync DB write + Async ML)
│   │   │   ├── teacher.py             # GET /teacher/tests/{id}/results (Cohort Psychometrics)
│   │   │   ├── concepts.py            # POST/DELETE /concepts/edges, GET /concepts/graph
│   │   │   ├── diagnosis.py           # GET /diagnosis/student/{id}/* (Student Dossier)
│   │   │   └── agent.py               # ReAct AI Agent Orchestration Routes
│   │   └── services/                  # Mathematical & Algorithmic Engines
│   │       ├── concept_graph.py       # Layer 1: NetworkX DAG Validation & Cycle Guards
│   │       ├── mastery_service.py     # Layer 2: Bayesian Knowledge Tracing (BKT Forward)
│   │       ├── dina_service.py        # Layer 3: Multi-Skill DINA + BFS Block Partitioning
│   │       ├── prediction_service.py  # Layer 4: Logistic Regression Slip/Guess Detector
│   │       ├── cognitive_analytics_service.py # Layer 5: Response-Time Effort (RTE) & Debt
│   │       └── agent_service.py       # ReAct AI Brain (Teaching Briefing & MCQ Synthesis)
│   ├── seed_dsa_diagnostic_test.py    # Automated seeder for sample tests and student assignment
│   └── requirements.txt               # Backend dependencies (FastAPI, SQLAlchemy, NetworkX, NumPy)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── PrerequisiteGraph.jsx  # SVG Force-Directed Prerequisite DAG Visualizer
│   │   ├── pages/
│   │   │   ├── TestBuilder.jsx        # Q-Matrix Multi-Skill Editor, DAG Controls & Bulk Import
│   │   │   ├── TakeTest.jsx           # Millisecond Client Telemetry Capture
│   │   │   ├── DiagnosticReport.jsx   # Student Dossier, Learning Roadmap & Effort Quadrants
│   │   │   └── TestResults.jsx        # Cohort Heatmap, Psychometrics & AI Intervention
│   │   └── App.jsx                    # Route Configuration
│   └── package.json                   # Frontend dependencies (React, Vite, Lucide-React, Axios)
├── advanced_ml_analytics_technical_report.tex # Formal Academic Technical Report (LaTeX)
├── advanced_ml_analytics_technical_report.pdf # Publication-Grade PDF with LaTeX Math
├── TEACHERS_PANEL_PRESENTATION_GUIDE.md       # Presentation Dossier & Defense Q&A
├── sample_15_tests_bulk_import.txt            # 15 Formatted Tests for 1-Click Bulk Import
└── ai_curriculum_15_topics_guide.md           # 15 Grounded AI Curriculum Topics
```

---

## 5. Summary of Teacher Capabilities

1. **Topological Curriculum Authoring:** Graph visualizer prevents circular dependencies and models strict prerequisites.
2. **Q-Matrix Multi-Skill Tagging:** Maps compound questions to multiple skills for fine-grained diagnosis.
3. **Cohort Mastery Heatmap:** Instant color-coded matrix of every student across all curriculum concepts.
4. **Item Psychometric Quality Badges:** Automatically identifies strong vs defective test questions using $r_{pb}$.
5. **Distractor Magnet Flags:** Discovers which wrong options acted as deceptive traps and reveals student misconceptions.
6. **Cohort Bottleneck Detection:** Isolates root blocker concepts dragging down class-wide pass rates.
7. **Agentic Closed-Loop Intervention:** AI writes teaching briefings and synthesizes editable remedial quizzes dispatched with one click to selected struggling students.

---

## 6. Installation & Local Execution

### Prerequisites
* Python 3.10+
* Node.js 18+
* PostgreSQL or Supabase account

### 1. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app.main:app --reload --port 8000
```
Backend API interactive documentation is available at `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
Frontend application will be accessible at `http://localhost:5173`.

---

## 7. Verification & Testing

To seed a complete sample test with multi-skill Q-matrix mappings, prerequisite DAG links, and automatic student assignment, execute:

```bash
# Inside backend directory with active virtual environment:
python seed_dsa_diagnostic_test.py
```

To run frontend production build validation:
```bash
cd frontend
npm run build
```

---

## 8. Academic Citation & Documentation References
* **De La Torre, J. (2009).** DINA Model and Parameter Estimation: A Didactic. *Journal of Educational and Behavioral Statistics*, 34(1), 115–130.
* **Corbett, A. T., & Anderson, J. R. (1995).** Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.
* **Wise, S. L., & Kong, X. (2005).** Response Time Effort: A New Measure of Examinee Motivation in Computer-Based Tests. *Applied Measurement in Education*, 18(2), 163–183.
* **Junker, B. W., & Sijtsma, K. (2001).** Cognitive Assessment Models with Few Assumptions: DINA and NIDA. *Applied Psychological Measurement*, 25(3), 258–272.
