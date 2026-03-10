"""
NAIVE RAG BASELINE — DLSU CpE Checklist AY 2022-2023
=====================================================
Dense Retrieval (all-MiniLM-L6-v2) + ChromaDB + LLM via HF Inference API

Stage 1: Hard-coded knowledge base (no database yet)
- Proves RAG pipeline works before adding DB complexity
- Establishes ROUGE-L, response time baseline scores, more testing basis to add
when llama is available
"""

from dotenv import load_dotenv
load_dotenv()

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# KNOWLEDGE BASE — sourced directly from DLSU CpE Checklist AY 2022-2023
# Each entry = one retrievable chunk stored in ChromaDB


DOCUMENTS = [

    # ── FIRST TERM ────────────────────────────────────────────────────────────
    {
        "id": "term1_overview",
        "text": "First Term courses: NSTP101 (National Service Training Program-General Orientation, 0 units), FNDMATH (Foundation in Math FOUN, 5 units), BASCHEM (Basic Chemistry, 3 units), BASPHYS (Basic Physics, 3 units), FNDSTAT (Foundation in Statistics FOUN, 3 units), GEARTAP (Art Appreciation 2A, 3 units). Total: 17 units. No prerequisites required for First Term.",
        "metadata": {"term": "1", "type": "overview"}
    },

    # ── SECOND TERM ───────────────────────────────────────────────────────────
    {
        "id": "term2_overview",
        "text": "Second Term courses: NSTPCW1 (National Service Training Program 1 2D, 3 units), GEMATMW (Mathematics in the Modern World 2A, 3 units), CALENG1 (Differential Calculus 1A, 3 units) requires FNDMATH as hard prerequisite, COEDISC (Computer Engineering as a Discipline 1E, 1 unit), PROLOGI (Programming Logic and Design Lecture 1E, 2 units), LBYCPA1 (Programming Logic and Design Laboratory 1E, 2 units) requires PROLOGI as co-requisite, LBYEC2A (Computer Fundamentals and Programming 1, 1 unit), GESTSOC (Science Technology and Society 2A, 3 units), GERIZAL (Life and Works of Rizal 2B, 3 units). Total: 18 units.",
        "metadata": {"term": "2", "type": "overview"}
    },
    {
        "id": "caleng1_prereq",
        "text": "CALENG1 (Differential Calculus) has FNDMATH as a hard prerequisite. Students must pass FNDMATH before enrolling in CALENG1.",
        "metadata": {"term": "2", "type": "prerequisite", "course": "CALENG1"}
    },
    {
        "id": "lbycpa1_coreq",
        "text": "LBYCPA1 (Programming Logic and Design Laboratory) requires PROLOGI as a co-requisite. Both PROLOGI and LBYCPA1 must be taken in the same term.",
        "metadata": {"term": "2", "type": "corequisite", "course": "LBYCPA1"}
    },

    # ── THIRD TERM ────────────────────────────────────────────────────────────
    {
        "id": "term3_overview",
        "text": "Third Term courses: NSTPCW2 (National Service Training Program 2 2D, 3 units) requires NSTPCW1 as hard prerequisite, LCLSONE (Lasallian Studies 1, 1 unit), SAS1000 (Student Affairs Service 1000 LS, 0 units), LASARE1 (Lasallian Recollection 1, 0 units), ENGPHYS (Physics for Engineers 1B, 3 units) requires CALENG1 as soft/hard prerequisite and BASPHYS, LBYPH1A (Physics for Engineers Laboratory 1B, 1 unit) requires ENGPHYS as co-requisite, CALENG2 (Integral Calculus 1A, 3 units) requires CALENG1 as hard prerequisite, LBYEC2A (Computer Fundamentals and Programming 2, 1 unit) requires LBYEC2A as hard prerequisite, LBYCPEI (Object Oriented Programming Laboratory 1E, 2 units) requires PROLOGI as hard prerequisite, GEPCOMM (Purposive Communications 2A, 3 units), LCFAITH (Faith Worth Living, 3 units), GELECSP (Social Science and Philosophy 2B, 3 units). Total: 19 units.",
        "metadata": {"term": "3", "type": "overview"}
    },
    {
        "id": "engphys_prereq",
        "text": "ENGPHYS (Physics for Engineers) requires CALENG1 as a soft/hard prerequisite and BASPHYS. LBYPH1A (Physics for Engineers Laboratory) requires ENGPHYS as a co-requisite.",
        "metadata": {"term": "3", "type": "prerequisite", "course": "ENGPHYS"}
    },
    {
        "id": "caleng2_prereq",
        "text": "CALENG2 (Integral Calculus) requires CALENG1 as a hard prerequisite. Students must pass CALENG1 before enrolling in CALENG2.",
        "metadata": {"term": "3", "type": "prerequisite", "course": "CALENG2"}
    },

    # ── FOURTH TERM ───────────────────────────────────────────────────────────
    {
        "id": "term4_overview",
        "text": "Fourth Term courses: CALENG3 (Differential Equations 1A, 3 units) requires CALENG2 as hard prerequisite, DATSRAL (Data Structures and Algorithms Lecture 1E, 1 unit) requires LBYCPEI as hard prerequisite, LBYCPA2 (Data Structures and Algorithms Laboratory 1E, 2 units) requires DATSRAL as co-requisite, DISCRMT (Discrete Mathematics 1E, 3 units) requires CALENG1 as hard prerequisite, FUNDCKT (Fundamentals of Electrical Circuits Lecture 1D, 3 units) requires ENGPHYS as hard prerequisite, LBYEC2M (Fundamentals of Electrical Circuits Lab 1D, 1 unit) requires FUNDCKT as co-requisite, ENGCHEM (Chemistry for Engineers 1B, 3 units) requires BASCHEM as hard prerequisite, LBYCH1A (Chemistry for Engineers Laboratory 1B, 1 unit) requires ENGCHEM as co-requisite, GEFTWEL (Physical Fitness and Wellness 2C, 2 units). Total: 19 units.",
        "metadata": {"term": "4", "type": "overview"}
    },
    {
        "id": "caleng3_prereq",
        "text": "CALENG3 (Differential Equations) requires CALENG2 as a hard prerequisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "CALENG3"}
    },
    {
        "id": "datsral_prereq",
        "text": "DATSRAL (Data Structures and Algorithms Lecture) requires LBYCPEI as a hard prerequisite. LBYCPA2 (Data Structures and Algorithms Laboratory) requires DATSRAL as a co-requisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "DATSRAL"}
    },
    {
        "id": "fundckt_prereq",
        "text": "FUNDCKT (Fundamentals of Electrical Circuits Lecture) requires ENGPHYS as a hard prerequisite. LBYEC2M (Fundamentals of Electrical Circuits Lab) requires FUNDCKT as a co-requisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "FUNDCKT"}
    },

    # ── FIFTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term5_overview",
        "text": "Fifth Term courses: ENGDATA (Engineering Data Analysis 1A, 3 units) requires CALENG2 or FNDSTAT as soft/hard prerequisite, NUMMETS (Numerical Methods 1E, 3 units) requires CALENG3 as hard prerequisite, FUNDLEC (Fundamentals of Electronic Circuits Lecture 1D, 3 units) requires FUNDCKT as hard prerequisite, LBYCPC2 (Fundamentals of Electronic Circuits Laboratory 1D, 1 unit) requires FUNDLEC as co-requisite, SOFDESG (Software Design Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPD2 (Software Design Laboratory 1E, 1 unit) requires SOFDESG as co-requisite, ENGENVI (Environmental Science and Engineering, 3 units) requires ENGCHEM as hard prerequisite, GEDANCE (Physical Fitness and Wellness in Dance 2C, 2 units), SAS2000 (Student Affairs Series 2, 0 units). Total: 19 units.",
        "metadata": {"term": "5", "type": "overview"}
    },
    {
        "id": "sofdesg_prereq",
        "text": "SOFDESG (Software Design Lecture) requires LBYCPA2 as a hard prerequisite. LBYCPD2 (Software Design Laboratory) requires SOFDESG as a co-requisite.",
        "metadata": {"term": "5", "type": "prerequisite", "course": "SOFDESG"}
    },
    {
        "id": "fundlec_prereq",
        "text": "FUNDLEC (Fundamentals of Electronic Circuits Lecture) requires FUNDCKT as a hard prerequisite. LBYCPC2 (Fundamentals of Electronic Circuits Laboratory) requires FUNDLEC as a co-requisite.",
        "metadata": {"term": "5", "type": "prerequisite", "course": "FUNDLEC"}
    },

    # ── SIXTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term6_overview",
        "text": "Sixth Term courses: LCLSTWO (Lasallian Studies 2, 1 unit), LASARE2 (Lasallian Recollection 2, 0 units), MXSIGFN (Fundamentals of Mixed Signals and Sensors 1E, 3 units) requires FUNDLEC as hard prerequisite, LOGDSGN (Logic Circuits and Design Lecture 1E, 3 units) requires FUNDLEC as hard prerequisite, LBYCPG3 (Logic Circuits and Design Laboratory 1E, 1 unit) requires LOGDSGN as co-requisite, FDCNSYS (Feedback and Control Systems 1E, 3 units) requires NUMMETS as hard prerequisite, LBYCPC3 (Feedback and Control System Laboratory 1E, 1 unit) requires FDCNSYS as co-requisite, LBYME1C (Computer-Aided Drafting CAD for ECE and CpE 1C, 1 unit), GELACAH (Arts and Humanities 2B, 3 units), GESPORT (Physical Fitness and Wellness in Individual Sports 2C, 2 units). Total: 17 units.",
        "metadata": {"term": "6", "type": "overview"}
    },
    {
        "id": "logdsgn_prereq",
        "text": "LOGDSGN (Logic Circuits and Design Lecture) requires FUNDLEC as a hard prerequisite. LBYCPG3 (Logic Circuits and Design Laboratory) requires LOGDSGN as a co-requisite.",
        "metadata": {"term": "6", "type": "prerequisite", "course": "LOGDSGN"}
    },
    {
        "id": "fdcnsys_prereq",
        "text": "FDCNSYS (Feedback and Control Systems) requires NUMMETS as a hard prerequisite. LBYCPC3 (Feedback and Control System Laboratory) requires FDCNSYS as a co-requisite.",
        "metadata": {"term": "6", "type": "prerequisite", "course": "FDCNSYS"}
    },

    # ── SEVENTH TERM ──────────────────────────────────────────────────────────
    {
        "id": "term7_overview",
        "text": "Seventh Term courses: GEETHIC (Ethics 2A, 3 units), MICROS (Microprocessors Lecture 1E, 3 units) requires LOGDSGN as hard prerequisite, LBYCPA3 (Microprocessors Laboratory 1E, 1 unit) requires MICROS as co-requisite, LBYCPB3 (Computer Engineering Drafting and Design Laboratory 1E, 1 unit) requires FUNDLEC and LOGDSGN as hard/hard prerequisite, LBYEC3B (Intelligent Systems for Engineering, 1 unit) requires LBYEC2A and ENGDATA as hard/hard prerequisite, LBYCPF2 (Introduction to HDL Laboratory 1E, 1 unit) requires FUNDLEC as hard prerequisite, DIGDACM (Data and Digital Communications 1E, 3 units) requires FUNDLEC as hard prerequisite, GETEAMS (Physical Fitness and Wellness in Team Sports 2C, 2 units), LBYCPG2 (Basic Computer Systems Administration, 1 unit). Total: 16 units.",
        "metadata": {"term": "7", "type": "overview"}
    },
    {
        "id": "micros_prereq",
        "text": "MICROS (Microprocessors Lecture) requires LOGDSGN as a hard prerequisite. LBYCPA3 (Microprocessors Laboratory) requires MICROS as a co-requisite.",
        "metadata": {"term": "7", "type": "prerequisite", "course": "MICROS"}
    },

    # ── EIGHTH TERM ───────────────────────────────────────────────────────────
    {
        "id": "term8_overview",
        "text": "Eighth Term courses: CSYSARC (Computer Architecture and Organization Lecture 1E, 3 units) requires MICROS as hard prerequisite, LBYCPD3 (Computer Architecture and Organization Laboratory 1E, 1 unit) requires CSYSARC as co-requisite, EMBDSYS (Embedded Systems Lecture 1E, 3 units) requires MICROS as hard prerequisite, LBYCPM3 (Embedded Systems Laboratory 1E, 1 unit) requires EMBDSYS as co-requisite, LBYCPG3 (Online Technologies Laboratory, 1 unit), GELECST (Science and Technology 2B, 3 units), REMETHS (Methods of Research for CpE 1E, 3 units) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites, OPESSYS (Operating Systems Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPO1 (Operating Systems Laboratory 1E, 1 unit) requires OPESSYS as co-requisite. Total: 8 units.",
        "metadata": {"term": "8", "type": "overview"}
    },
    {
        "id": "embdsys_prereq",
        "text": "EMBDSYS (Embedded Systems Lecture) requires MICROS as a hard prerequisite. LBYCPM3 (Embedded Systems Laboratory) requires EMBDSYS as a co-requisite.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "EMBDSYS"}
    },
    {
        "id": "csysarc_prereq",
        "text": "CSYSARC (Computer Architecture and Organization Lecture) requires MICROS as a hard prerequisite. LBYCPD3 (Computer Architecture and Organization Laboratory) requires CSYSARC as a co-requisite.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "CSYSARC"}
    },
    {
        "id": "remeths_prereq",
        "text": "REMETHS (Methods of Research for CpE) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "REMETHS"}
    },

    # ── NINTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term9_overview",
        "text": "Ninth Term courses: LCLSTRI (Lasallian Studies 3, 1 unit), LCASEAN (The Filipino and ASEAN, 3 units), LASARE3 (Lasallian Recollection 3, 0 units), DSIGPRO (Digital Signal Processing Lecture 1E, 3 units) requires FDCNSYS and EMBDSYS as hard/soft prerequisite, LBYCPA4 (Digital Signal Processing Laboratory 1E, 1 unit) requires DSIGPRO as co-requisite, OCHESAF (Basic Occupational Health and Safety 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4A (CpE Practice and Design 1 1E, 1 unit) requires EMBDSYS and REMETHS as hard prerequisites, CPEPRAC (CpE Laws and Professional Practice 1E, 2 units) requires EMBDSYS as hard prerequisite, CPECOG1 (CpE Elective 1 Lecture 1F, 2 units) requires EMBDSYS and THSCP4A as hard/co prerequisite, LBYCPF3 (CpE Elective 1 Laboratory 1F, 1 unit) requires CPECOG1 as co-requisite. Total: 16 units.",
        "metadata": {"term": "9", "type": "overview"}
    },
    {
        "id": "thscp4a_prereq",
        "text": "THSCP4A (CpE Practice and Design 1) requires both EMBDSYS and REMETHS as hard prerequisites. This is a capstone/thesis preparation course.",
        "metadata": {"term": "9", "type": "prerequisite", "course": "THSCP4A"}
    },
    {
        "id": "dsigpro_prereq",
        "text": "DSIGPRO (Digital Signal Processing Lecture) requires FDCNSYS as a hard prerequisite and EMBDSYS as a soft prerequisite. LBYCPA4 (Digital Signal Processing Laboratory) requires DSIGPRO as a co-requisite.",
        "metadata": {"term": "9", "type": "prerequisite", "course": "DSIGPRO"}
    },

    # ── TENTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term10_overview",
        "text": "Tenth Term courses: LCENWRD (Encountering the Word in the World, 3 units), EMERTEC (Emerging Technologies in CpE 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4B (CpE Practice and Design 2 1E, 1 unit) requires THSCP4A as hard prerequisite, ENGTREP (Technopreneurship 101 1C, 3 units) requires EMBDSYS as hard prerequisite, CONETSC (Computer Networks and Security Lecture 1E, 3 units) requires DIGDACM as hard prerequisite, LBYCPB4 (Computer Networks and Security Laboratory 1E, 1 unit) requires CONETSC as co-requisite, CPECAPS (Operational Technologies, 2 units) requires LBYCPB3 and LBYCPB4 as co/co requisite, CPECOG2 (CpE Elective 2 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPH3 (CpE Elective 2 Laboratory 1F, 1 unit) requires CPECOG2 as co-requisite, SAS3000 (Student Affairs Series 3, 0 units) requires SAS2000 as hard prerequisite. Total units vary.",
        "metadata": {"term": "10", "type": "overview"}
    },
    {
        "id": "conetsc_prereq",
        "text": "CONETSC (Computer Networks and Security Lecture) requires DIGDACM as a hard prerequisite. LBYCPB4 (Computer Networks and Security Laboratory) requires CONETSC as a co-requisite.",
        "metadata": {"term": "10", "type": "prerequisite", "course": "CONETSC"}
    },
    {
        "id": "thscp4b_prereq",
        "text": "THSCP4B (CpE Practice and Design 2) requires THSCP4A as a hard prerequisite. This is the second part of the capstone/thesis sequence.",
        "metadata": {"term": "10", "type": "prerequisite", "course": "THSCP4B"}
    },

    # ── ELEVENTH TERM ─────────────────────────────────────────────────────────
    {
        "id": "term11_overview",
        "text": "Eleventh Term: PRCGECP (Practicum for CpE 1E, 3 units) requires REMETHS as hard prerequisite. Total: 3 units. This is the practicum/internship term.",
        "metadata": {"term": "11", "type": "overview"}
    },

    # ── TWELFTH TERM ──────────────────────────────────────────────────────────
    {
        "id": "term12_overview",
        "text": "Twelfth Term courses: GERPHIS (Readings in the Philippine History 2A, 3 units), GEWORLD (The Contemporary World 2A, 3 units), THSCP4C (CpE Practice and Design 3 1E, 1 unit) requires THSCP4B as hard prerequisite, CPECOG3 (CpE Elective 3 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPC4 (CpE Elective 3 Laboratory 1F, 1 unit) requires CPECOG3 as co-requisite, CPETRIP (Seminars and Field Trips for CpE 1E, 1 unit) requires EMBDSYS and CPECAPS as hard prerequisites, ECNOMIC (Engineering Economics for CpE 1C, 3 units) requires CALENG1 as soft prerequisite, ENGMANA (Engineering Management, 2 units) requires CALENG1 as soft prerequisite, GEUSELF (Understanding the Self 2A, 3 units). Total: 19 units.",
        "metadata": {"term": "12", "type": "overview"}
    },
    {
        "id": "thscp4c_prereq",
        "text": "THSCP4C (CpE Practice and Design 3) requires THSCP4B as a hard prerequisite. This is the final part of the capstone/thesis sequence. The full thesis sequence is THSCP4A → THSCP4B → THSCP4C.",
        "metadata": {"term": "12", "type": "prerequisite", "course": "THSCP4C"}
    },

    # ── GENERAL POLICIES ──────────────────────────────────────────────────────
    {
        "id": "prereq_legend",
        "text": "Prerequisite Legend: H = Hard Pre-Requisite (must be passed before enrolling), S = Soft Pre-Requisite (should be passed; not following will cause the course to be INVALIDATED), C = Co-Requisite (must be taken in the same term). This checklist is for freshmen who started AY 2022-2023.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "checklist_warning",
        "text": "Students should not enroll without passing their respective hard prerequisites. Taking a soft pre-requisite course without passing it will cause the course to be INVALIDATED. This checklist is tentative and subject to change.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "thesis_sequence",
        "text": "The CpE thesis/capstone sequence is: THSCP4A (Term 9, requires EMBDSYS + REMETHS) → THSCP4B (Term 10, requires THSCP4A) → THSCP4C (Term 12, requires THSCP4B). Students must complete this sequence to graduate.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "nstp_sequence",
        "text": "The NSTP sequence is: NSTP101 (Term 1, General Orientation) → NSTPCW1 (Term 2) → NSTPCW2 (Term 3, requires NSTPCW1 as hard prerequisite).",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "lasallian_sequence",
        "text": "The Lasallian Studies sequence is: LCLSONE (Term 3) → LCLSTWO (Term 6) → LCLSTRI (Term 9). Lasallian Recollections: LASARE1 (Term 3), LASARE2 (Term 6), LASARE3 (Term 9).",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "sas_sequence",
        "text": "Student Affairs Series: SAS1000 (Term 3, 0 units) → SAS2000 (Term 5, 0 units) → SAS3000 (Term 10, 0 units, requires SAS2000 as hard prerequisite).",
        "metadata": {"term": "all", "type": "policy"}
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES — ground truths written from the checklist above
# These are your SO1 / SO3 evaluation inputs
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "question": "What are the prerequisites for CALENG2?",
        "ground_truth": "CALENG2 requires CALENG1 as a hard prerequisite. Students must pass CALENG1 before enrolling in CALENG2."
    },
    {
        "question": "What are the prerequisites for DATSRAL?",
        "ground_truth": "DATSRAL requires LBYCPEI as a hard prerequisite."
    },
    {
        "question": "What is the co-requisite of LBYCPA2?",
        "ground_truth": "LBYCPA2 is the co-requisite of DATSRAL. Both must be taken in the same term."
    },
    {
        "question": "What courses are in the seventh term?",
        "ground_truth": "Seventh Term includes GEETHIC, MICROS, LBYCPA3, LBYCPB3, LBYEC3B, LBYCPF2, DIGDACM, GETEAMS, and LBYCPG2. Total of 16 units."
    },
    {
        "question": "What does H mean in the prerequisite legend?",
        "ground_truth": "H means Hard Pre-Requisite. Students must pass the hard prerequisite course before enrolling in the next course."
    },
    {
        "question": "What happens if I take a course without passing its soft prerequisite?",
        "ground_truth": "If a student takes a course without passing its soft prerequisite, the course will be INVALIDATED."
    },
    {
        "question": "What are the prerequisites for THSCP4A?",
        "ground_truth": "THSCP4A requires both EMBDSYS and REMETHS as hard prerequisites."
    },
    {
        "question": "What is the thesis sequence for CpE students?",
        "ground_truth": "The thesis sequence is THSCP4A in Term 9, then THSCP4B in Term 10, then THSCP4C in Term 12."
    },
    {
        "question": "What are the prerequisites for LOGDSGN?",
        "ground_truth": "LOGDSGN requires FUNDLEC as a hard prerequisite."
    },
    {
        "question": "What are the prerequisites for EMBDSYS?",
        "ground_truth": "EMBDSYS requires MICROS as a hard prerequisite."
    },
    {
        "question": "What is the prerequisite for MICROS?",
        "ground_truth": "MICROS requires LOGDSGN as a hard prerequisite."
    },
    {
        "question": "What term is PRCGECP taken and what is its prerequisite?",
        "ground_truth": "PRCGECP is taken in the Eleventh Term and requires REMETHS as a hard prerequisite."
    },
    {
        "question": "What is the co-requisite relationship between SOFDESG and LBYCPD2?",
        "ground_truth": "LBYCPD2 is the laboratory co-requisite of SOFDESG. Both must be taken in the same term."
    },
    {
        "question": "What prerequisites does REMETHS require?",
        "ground_truth": "REMETHS requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites."
    },
    {
        "question": "Can I take FUNDLEC without passing FUNDCKT?",
        "ground_truth": "No. FUNDLEC requires FUNDCKT as a hard prerequisite and must be passed before enrolling."
    },
]


#  BUILD CHROMADB VECTOR STORE


def build_vector_store():
    print("Building ChromaDB vector store with all-MiniLM-L6-v2...")

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
        # 384-dim embeddings
    )

    client = chromadb.Client()  # in-memory for now, swap to PersistentClient later
    collection = client.create_collection(
        name="dlsu_cpe_checklist",
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS],
    )

    print(f"Stored {len(DOCUMENTS)} document chunks\n")
    return collection



# STEP 2: RETRIEVAL FUNCTION


def retrieve(collection, query: str, top_k: int = 3) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    # Flatten retrieved chunks into one context string
    chunks = results["documents"][0]
    return "\n\n".join(chunks)



#  GENERATION FUNCTION

def generate(client: InferenceClient, model_id: str, context: str, question: str) -> str:
    response = client.chat_completion(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an academic adviser for DLSU Computer Engineering students. "
                    "Use ONLY the provided context to answer questions accurately. "
                    "If the answer is not in the context, say: "
                    "'I don't have that information — please consult your adviser.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=200,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()



# EVALUATION LOOP


def run_evaluation(model_id: str):
    print(f"\n{'='*65}")
    print(f"MODEL: {model_id}")
    print(f"{'='*65}\n")

    # Setup
    collection = build_vector_store()
    hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    retrieval_times = []
    generation_times = []
    total_times = []
    results_log = []

    for i, test in enumerate(TEST_CASES):
        print(f"Q{i+1}: {test['question']}")

        total_start = time.time()

        # Retrieval
        t0 = time.time()
        context = retrieve(collection, test["question"], top_k=3)
        retrieval_time = time.time() - t0

        # Generation
        t1 = time.time()
        answer = generate(hf_client, model_id, context, test["question"])
        generation_time = time.time() - t1

        total_time = time.time() - total_start

        # Score
        score = rouge.score(test["ground_truth"], answer)
        rouge_l = score["rougeL"].fmeasure

        scores.append(rouge_l)
        retrieval_times.append(retrieval_time)
        generation_times.append(generation_time)
        total_times.append(total_time)

        print(f"  Answer:     {answer[:120]}...")
        print(f"  ROUGE-L:    {rouge_l:.3f}")
        print(f"  Retrieval:  {retrieval_time:.2f}s")
        print(f"  Generation: {generation_time:.2f}s")
        print(f"  Total:      {total_time:.2f}s {'[OK]' if total_time < 5 else '[OVER 5s]'}\n")

        results_log.append({
            "question": test["question"],
            "ground_truth": test["ground_truth"],
            "answer": answer,
            "rouge_l": rouge_l,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
        })

    # Summary
    avg_rouge    = sum(scores) / len(scores)
    avg_ret_time = sum(retrieval_times) / len(retrieval_times)
    avg_gen_time = sum(generation_times) / len(generation_times)
    avg_total    = sum(total_times) / len(total_times)
    under_5s     = sum(1 for t in total_times if t < 5) / len(total_times) * 100

    print(f"\n{'='*65}")
    print(f"SUMMARY — {model_id}")
    print(f"{'='*65}")
    print(f"Avg ROUGE-L:         {avg_rouge:.3f}")
    print(f"Avg Retrieval Time:  {avg_ret_time:.2f}s")
    print(f"Avg Generation Time: {avg_gen_time:.2f}s")
    print(f"Avg Total Time:      {avg_total:.2f}s")
    print(f"Under 5s (SO4):      {under_5s:.0f}% of queries")
    print(f"{'='*65}\n")

    return {
        "model": model_id,
        "avg_rouge_l": avg_rouge,
        "avg_retrieval_time": avg_ret_time,
        "avg_generation_time": avg_gen_time,
        "avg_total_time": avg_total,
        "pct_under_5s": under_5s,
        "detail": results_log,
    }



# MAIN — test one model at a time, add more to MODELS list as needed


MODELS_TO_TEST = [
    "meta-llama/Llama-3.1-8B-Instruct",   # baseline
    "Qwen/Qwen2.5-7B-Instruct",          # uncomment to compare
    "mistralai/Mistral-7B-Instruct-v0.3",# uncomment to compare
]

if __name__ == "__main__":
    all_results = []

    for model in MODELS_TO_TEST:
        result = run_evaluation(model)
        all_results.append(result)

    # Final comparison table
    if len(all_results) > 1:
        print("\nFINAL MODEL COMPARISON")
        print(f"{'Model':<45} {'ROUGE-L':>8} {'Avg Time':>10} {'<5s':>6}")
        print("-" * 75)
        for r in sorted(all_results, key=lambda x: -x["avg_rouge_l"]):
            print(f"{r['model']:<45} {r['avg_rouge_l']:>8.3f} "
                  f"{r['avg_total_time']:>9.2f}s {r['pct_under_5s']:>5.0f}%")
