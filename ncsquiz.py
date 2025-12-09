#!/usr/bin/env python3
"""
Apni Disha - Hybrid NCS RIASEC Adaptive Chatbot (upgraded)

Key upgrades over original:
- Hybrid NCS metadata: LLM returns NCS-style metadata (ncs_code optional),
  job_family, skills_roadmap, salary_range (India-friendly).
- Improved normalization: per-trait normalized mean + trait confidence
  (based on number of questions asked).
- Career confidence computed from trait alignment and coverage.
- MCQ answers now influence secondary preference weights.
- Robust JSON parsing and richer output-schema.
"""

import json
import random
import os
from dotenv import load_dotenv
from time import sleep
from typing import Tuple, Dict, Any, List

# LangChain / Gemini imports (same as original environment)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ---------- Config ----------
QUESTIONS_FILE = "questions.json"
NUM_QUESTIONS_TO_ASK = 6
NUM_LLM_GENERATED_QUESTIONS = 5
MODEL_NAME = "gemini-2.0-flash"
API_ENV_VAR = "GOOGLE_API_KEY"
MIN_QUESTIONS_PER_TRAIT_FOR_HIGH_CONFIDENCE = 1  # with adaptive flow we'll scale confidence
# ----------------------------

# Load question bank
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    QUESTION_BANK = json.load(f)

TRAITS = ["R", "I", "A", "S", "E", "C"]

# Validate
for t in TRAITS:
    if t not in QUESTION_BANK:
        raise ValueError(f"Trait {t} missing in {QUESTIONS_FILE}")

# Input -> contribution
SCORE_MAP = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

# Runtime data
raw_scores = {t: 0.0 for t in TRAITS}
questions_asked = {t: [] for t in TRAITS}
qa_history: List[Tuple[str, str, Any]] = []  # (trait, question, rating_or_choice)

# Secondary preference weights (created/updated from MCQ answers)
# Each weight in [-1, 1] where positive means leaning toward that pole.
secondary_weights = {
    "analytical_vs_creative": 0.0,
    "solo_vs_team": 0.0,
    "structured_vs_flexible": 0.0
}

chat_history_messages = [
    SystemMessage(content="You are a helpful career-counselor assistant. "
                          "When asked to recommend careers, respond in valid JSON following the schema requested.")
]

# instantiate LLM
api_key = os.getenv(API_ENV_VAR)
if not api_key:
    raise EnvironmentError(f"Please set {API_ENV_VAR} in your .env file.")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key,
    temperature=0.2
)

# Small fallback hybrid NCS mapping for quick local enrichment (keeps implementation fast)
# This is intentionally small — the LLM will be asked to provide richer NCS-style metadata.
FALLBACK_NCS_MAP = {
    "Software Developer": {
        "ncs_code": "2520",
        "job_family": "Information Technology",
        "common_skills": ["programming", "algorithms", "debugging"]
    },
    "Data Analyst": {
        "ncs_code": "2410",
        "job_family": "Data & Analytics",
        "common_skills": ["statistics", "sql", "data-visualization"]
    },
    "Mechanical Engineer": {
        "ncs_code": "2140",
        "job_family": "Engineering",
        "common_skills": ["mechanics", "CAD", "manufacturing"]
    },
    # Add more fallback mappings as needed
}


def pick_next_question():
    """
    Adaptive selection: choose trait(s) with the fewest questions asked.
    """
    counts = {t: len(questions_asked[t]) for t in TRAITS}
    min_count = min(counts.values())
    candidates = [t for t, c in counts.items() if c == min_count]
    chosen_trait = random.choice(candidates)
    available = [q for q in QUESTION_BANK[chosen_trait] if q not in questions_asked[chosen_trait]]
    if not available:
        available = QUESTION_BANK[chosen_trait][:]
    question = random.choice(available)
    return chosen_trait, question


def get_user_rating(prompt_text):
    while True:
        ans = input(prompt_text + " (1-5): ").strip()
        if ans.isdigit():
            val = int(ans)
            if 1 <= val <= 5:
                return val
        print("Invalid input — please enter an integer between 1 and 5.")


def normalize_scores_and_confidence() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For each trait:
      - normalized score = mean contribution (0..1) if any questions asked else 0.5
      - confidence = min(1.0, questions_asked_count / target_count) scaled to [0.2..1.0]
    Returns (normalized_scores, trait_confidence)
    """
    normalized = {}
    confidence = {}
    # target_count chosen as ceil(NUM_QUESTIONS_TO_ASK / len(TRAITS)) minimal expectation
    target_count = max(1, (NUM_QUESTIONS_TO_ASK // len(TRAITS)) + 1)

    for t in TRAITS:
        n = len(questions_asked[t])
        if n == 0:
            normalized[t] = 0.5
            confidence[t] = 0.2  # low confidence when no data
        else:
            mean = raw_scores[t] / n
            normalized[t] = round(mean, 4)
            # confidence scales with coverage; we ensure minimum 0.3
            conf = min(1.0, n / target_count)
            # scale conf to [0.3, 1.0] to avoid zero-confidence
            scaled_conf = round(0.3 + 0.7 * conf, 4)
            confidence[t] = scaled_conf
    return normalized, confidence


def build_recommendation_prompt(qa_history: List[Tuple[str, str, Any]],
                                normalized_scores: Dict[str, float],
                                trait_confidence: Dict[str, float],
                                secondary_weights: Dict[str, float]) -> str:
    """
    Build a precise instruction for Gemini to return 2-3 career recommendations with
    Hybrid NCS metadata and enhanced schema.
    """
    qa_lines = []
    for idx, (trait, question, rating) in enumerate(qa_history, start=1):
        qa_lines.append(f"{idx}. Trait: {trait} | Q: {question} | Ans: {rating}")

    human_text = f"""
You are an expert career counselor and must output ONLY valid JSON following the EXACT schema below.

CONTEXT:
- This system uses Holland's RIASEC model: R, I, A, S, E, C.
- We provide normalized per-trait scores (0..1) and per-trait confidence (0..1).
- Secondary weights (analytical_vs_creative, solo_vs_team, structured_vs_flexible) are provided in [-1,1].

INPUT DATA:
User Q&A (ordered):
{json.dumps(qa_lines, indent=2)}

Normalized RIASEC scores:
{json.dumps(normalized_scores, indent=2)}

Per-trait confidence:
{json.dumps(trait_confidence, indent=2)}

Secondary preference weights:
{json.dumps(secondary_weights, indent=2)}

OUTPUT SCHEMA (MUST FOLLOW EXACTLY):
{{
  "recommendations": [
    {{
      "career": "<career name>",
      "ncs_code": "<optional NCS code or empty string>",
      "job_family": "<job family / sector>",
      "confidence": <float 0..1>,             // overall confidence for this recommendation
      "reason": "<short reason (1-2 sentences) tied to RIASEC scores & secondary prefs>",
      "stream": "<science|commerce|arts|other>",
      "salary_range_in_inr_per_annum": "<low-high or approximate>",
      "degrees": [
        {{
          "degree": "<general degree name>",
          "specializations": ["<spec1>", "<spec2>", "<spec3>"]
        }}
      ],
      "skills_roadmap": ["<skill1>", "<skill2>", "<skill3>"],   // practical next-steps for 6-12 months
      "ncs_recommended_training_or_certifications": ["<cert1>", "<cert2>"] // optional
    }}
  ]
}}

INSTRUCTIONS:
- Generate exactly 2 or 3 recommendation objects.
- Career names should be India-contextual (e.g., 'Software Developer', 'Data Analyst', 'Mechanical Engineer').
- Try to include an NCS-style code if you can; otherwise set it to an empty string.
- Confidence must combine normalized trait alignment and trait confidences; provide realistic values 0..1.
- Salary ranges should be approximate and India-focused (INR per annum).
- Degrees and specializations must be realistic and commonly offered in India.
- Skills roadmap must be concrete, actionable (6-12 months).
- Output JSON only. No explanations or extra text.

Now analyze the inputs above and output recommendations.
"""
    return human_text


def ask_llm_for_recommendations(qa_history, normalized_scores, trait_confidence, secondary_weights):
    human_prompt = build_recommendation_prompt(qa_history, normalized_scores, trait_confidence, secondary_weights)
    chat_msgs = chat_history_messages + [HumanMessage(content=human_prompt)]

    print("\nContacting LLM for career recommendations (JSON)...\n")
    response = llm.invoke(chat_msgs)

    text = getattr(response, "content", None)
    if text is None:
        try:
            text = response.generations[0][0].text
        except Exception:
            text = str(response)

    text_str = text.strip()

    # Attempt JSON extraction
    try:
        data = json.loads(text_str)
        return data, text_str
    except Exception:
        start = text_str.find("{")
        end = text_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text_str[start:end+1]
            try:
                data = json.loads(snippet)
                return data, text_str
            except Exception:
                pass
    return None, text_str


def generate_llm_mcq_questions(qa_history, num_questions=NUM_LLM_GENERATED_QUESTIONS):
    """
    Ask LLM to generate MCQs that refine secondary preferences.
    """
    qa_lines = []
    for idx, (trait, question, rating) in enumerate(qa_history, start=1):
        qa_lines.append(f"{idx}. Trait={trait} | Q=\"{question}\" | Ans={rating}")

    human_text = (
        "You are an expert in psychometric assessment. Based on the user's previous "
        "answers below, generate EXACTLY {n} JSON-formatted multiple-choice questions "
        "with 4 options (A-D). Each option should be designed so that selection "
        "maps to a meaningful secondary preference (analytical_vs_creative, solo_vs_team, structured_vs_flexible).\n\n"
        "Output JSON ONLY in this schema:\n"
        "{{\n"
        '  "questions": [\n'
        '    {{"question": "<text>", "options": {"A":"...","B":"...","C":"...","D":"..."}} }}\n'
        "  ]\n"
        "}}\n\n"
        "User Q&A history:\n"
        + "\n".join(qa_lines)
    ).format(n=num_questions)

    chat_msgs = chat_history_messages + [HumanMessage(content=human_text)]
    print("\nGenerating MCQ questions using LLM...\n")
    response = llm.invoke(chat_msgs)

    text = getattr(response, "content", "").strip()
    try:
        data = json.loads(text)
        return data.get("questions", [])
    except:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end+1])
                return data.get("questions", [])
            except:
                pass
    print("LLM MCQ JSON parse failed. Raw output:\n", text)
    return []


def run_mcq_stage(mcq_list, qa_history):
    """
    Ask MCQs; map choices into updates to secondary_weights.
    We'll use a simple mapping strategy:
      - Each option A-D has metadata embedded by us: (analytical,creative),(solo,team),(structured,flexible) influence.
    Since LLM generated questions won't return mapping, we'll use heuristic mapping:
      A -> analytical, solo, structured
      B -> analytical, team, structured
      C -> creative, team, flexible
      D -> creative, solo, flexible
    These heuristics are deliberately simple and can be improved later.
    """
    print("\n=== LLM-Generated MCQ Stage ===")
    print(f"You will now answer {len(mcq_list)} additional questions.\n")

    for i, qdata in enumerate(mcq_list, start=1):
        print(f"\nMCQ {i}: {qdata['question']}")
        for opt_key, opt_text in qdata["options"].items():
            print(f"  {opt_key}. {opt_text}")

        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ["A", "B", "C", "D"]:
                break
            print("Invalid choice — select A, B, C, or D.")

        qa_history.append(("MCQ", qdata['question'], choice))
        chat_history_messages.append(HumanMessage(content=f"User answered MCQ: Q='{qdata['question']}' | Choice={choice}"))

        # update secondary_weights heuristic
        delta = 0.15  # influence magnitude per MCQ (tunable)
        if choice == "A":
            secondary_weights["analytical_vs_creative"] += delta
            secondary_weights["solo_vs_team"] += delta
            secondary_weights["structured_vs_flexible"] += delta
        elif choice == "B":
            secondary_weights["analytical_vs_creative"] += delta
            secondary_weights["solo_vs_team"] -= delta
            secondary_weights["structured_vs_flexible"] += delta
        elif choice == "C":
            secondary_weights["analytical_vs_creative"] -= delta
            secondary_weights["solo_vs_team"] -= delta
            secondary_weights["structured_vs_flexible"] -= delta
        elif choice == "D":
            secondary_weights["analytical_vs_creative"] -= delta
            secondary_weights["solo_vs_team"] += delta
            secondary_weights["structured_vs_flexible"] -= delta

        # clamp weights
        for k in secondary_weights:
            secondary_weights[k] = max(-1.0, min(1.0, round(secondary_weights[k], 4)))


def career_confidence_from_alignment(recommended_traits: List[Tuple[str, float]],
                                     trait_confidence: Dict[str, float],
                                     coverage_factor: float) -> float:
    """
    recommended_traits: list of (trait_letter, trait_score) for the career
    trait_confidence: per-trait confidence [0..1]
    coverage_factor: how many traits had questions asked (0..1)
    Compute a blended confidence metric
    """
    # base = weighted average of trait score * trait_confidence
    numer = 0.0
    denom = 0.0
    for t, s in recommended_traits:
        conf = trait_confidence.get(t, 0.3)
        numer += s * conf
        denom += conf
    if denom == 0:
        base = 0.5
    else:
        base = numer / denom
    # blend with coverage_factor (gives benefit to well-covered profiles)
    final = 0.6 * base + 0.4 * coverage_factor
    return round(max(0.0, min(1.0, final)), 4)


def enrich_with_fallback_ncs(career_name: str) -> Dict[str, Any]:
    """
    Provide small fallback metadata quickly (so system can respond fast).
    LLM will still be asked to return richer metadata; this is local best-effort fallback.
    """
    return FALLBACK_NCS_MAP.get(career_name, {"ncs_code": "", "job_family": "", "common_skills": []})


def main():
    print("=== Apni Disha: Adaptive RIASEC Test (Hybrid NCS) ===")
    print(f"You will be asked {NUM_QUESTIONS_TO_ASK} adaptive questions.")
    print("Rate each question from 1 (low/negative) to 5 (high/positive).")
    input("Press Enter to start...")

    asked = 0
    while asked < NUM_QUESTIONS_TO_ASK:
        trait, question = pick_next_question()
        print(f"\nQuestion {asked+1}: ({trait}) {question}")
        rating = get_user_rating("Your rating")

        contribution = SCORE_MAP.get(rating, 0.5)
        raw_scores[trait] += contribution
        questions_asked[trait].append(question)
        qa_history.append((trait, question, rating))
        chat_history_messages.append(HumanMessage(content=f"User answered: Trait={trait} | Q='{question}' | Rating={rating}"))

        asked += 1

    print("\n--- Stage 1 (RIASEC) Complete ---")

    # Generate MCQs using LLM
    mcq_questions = generate_llm_mcq_questions(qa_history, NUM_LLM_GENERATED_QUESTIONS)

    if mcq_questions:
        run_mcq_stage(mcq_questions, qa_history)
    else:
        print("Skipping MCQ stage due to generation error.")

    print("\n--- All Questions Completed ---")
    normalized, trait_confidence = normalize_scores_and_confidence()

    print("\nNormalized RIASEC scores (0-1) and trait confidence (0-1):")
    for t in TRAITS:
        print(f" {t}: score={normalized[t]}  conf={trait_confidence[t]}")

    # Compute coverage factor: fraction of traits with at least one question asked
    coverage = sum(1 for t in TRAITS if len(questions_asked[t]) > 0) / len(TRAITS)
    coverage = round(coverage, 4)

    # Ask LLM for recommendations (it will produce NCS-style metadata too)
    json_result, raw_text = ask_llm_for_recommendations(qa_history, normalized, trait_confidence, secondary_weights)

    print("\nLLM returned (raw):\n")
    print(raw_text)
    print("\nParsed JSON (if parse succeeded):\n")
    if json_result is None:
        print("Failed to parse JSON. Raw LLM text shown above.")
        # still attempt to provide fallback: use top 2 traits to suggest careers locally
        sorted_traits = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        top2 = [t for t, _ in sorted_traits[:2]]
        fallback_careers = []
        if "I" in top2 or "R" in top2:
            fallback_careers.append("Software Developer")
        if "S" in top2 or "E" in top2:
            fallback_careers.append("Sales / Business Development")
        if "A" in top2:
            fallback_careers.append("Graphic Designer")
        print("Fallback recommendations (local):", fallback_careers)
    else:
        # Post-process recommendations to add fallback NCS metadata and compute final confidence
        recs = json_result.get("recommendations", [])
        processed = []
        for rec in recs:
            career = rec.get("career", "Unknown")
            # Enrich with fallback NCS metadata if missing
            fallback = enrich_with_fallback_ncs(career)
            if not rec.get("ncs_code"):
                rec["ncs_code"] = fallback.get("ncs_code", "")
            if not rec.get("job_family"):
                rec["job_family"] = fallback.get("job_family", "")
            if "confidence" not in rec:
                # heuristics: match top traits
                # ask recommender LLM's reason may reference trait names; but we compute alignment roughly
                # We'll compute recommended_traits as top 2 RIASEC traits for career using normalized scores
                sorted_traits = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
                top_traits = sorted_traits[:2]
                recommended_traits = [(t, normalized[t]) for t, _ in top_traits]
                conf = career_confidence_from_alignment(recommended_traits, trait_confidence, coverage)
                rec["confidence"] = conf
            # ensure skill roadmap exists
            if "skills_roadmap" not in rec or not rec["skills_roadmap"]:
                rec["skills_roadmap"] = fallback.get("common_skills", [])[:3]
            processed.append(rec)

        # final output
        print(json.dumps({"recommendations": processed}, indent=2, ensure_ascii=False))

    print("\nYou can use the normalized RIASEC scores above to plot a chart (each value 0-1).")
    print("\nDone. Thank you!")


if __name__ == "__main__":
    main()