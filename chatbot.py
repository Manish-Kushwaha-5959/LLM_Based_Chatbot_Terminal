"""
Terminal RIASEC adaptive chatbot with LangChain + Google Gemini (gemini-2.0-flash).

Features:
- Loads questions from questions.json
- Adaptive question selection (chooses trait with fewest asked)
- Accepts user rating 1-5 (1=low/negative, 5=high/positive)
- Scoring per question maps 1..5 -> 0.0..1.0
- After 6 questions sends full Q&A and normalized RIASEC scores to Gemini
  and requests a JSON output with 2-3 career recommendations and reasons.
- Prints the returned JSON and also prints normalized RIASEC scores (0-1).
"""

import json
import random
import os
from dotenv import load_dotenv
from time import sleep

# LangChain / Gemini imports
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ---------- Config ----------
QUESTIONS_FILE = "questions.json"
NUM_QUESTIONS_TO_ASK = 6
NUM_LLM_GENERATED_QUESTIONS = 5   # you can change this anytime
MODEL_NAME = "gemini-2.0-flash"
API_ENV_VAR = "GOOGLE_API_KEY"  # ensure .env has this key
# ----------------------------

# load questions
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    QUESTION_BANK = json.load(f)

TRAITS = ["R", "I", "A", "S", "E", "C"]

# validate bank has keys
for t in TRAITS:
    if t not in QUESTION_BANK:
        raise ValueError(f"Trait {t} missing in {QUESTIONS_FILE}")

# scoring map for user input 1-5 -> contribution 0.0-1.0
SCORE_MAP = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

# runtime data structures
raw_scores = {t: 0.0 for t in TRAITS}           # sum of contributions per trait
questions_asked = {t: [] for t in TRAITS}       # list of questions asked per trait
qa_history = []                                 # ordered list of (trait, question, rating)
chat_history_messages = [                        # langchain messages (system + human/ai)
    SystemMessage(content="You are a helpful career-counselor assistant. "
                          "When asked to recommend careers, respond strictly in JSON format as instructed.")
]

# instantiate LLM
api_key = os.getenv(API_ENV_VAR)
if not api_key:
    raise EnvironmentError(f"Please set {API_ENV_VAR} in your .env file (e.g. {API_ENV_VAR}=...).")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key,
    temperature=0.2
)


def pick_next_question():
    """
    Adaptive selection:
    - choose the trait(s) with the fewest questions asked
    - break ties randomly
    - choose a question from that trait not asked before (if possible)
    """
    # find fewest asked count
    counts = {t: len(questions_asked[t]) for t in TRAITS}
    min_count = min(counts.values())
    candidate_traits = [t for t, c in counts.items() if c == min_count]

    chosen_trait = random.choice(candidate_traits)

    # choose unused question if possible
    available = [q for q in QUESTION_BANK[chosen_trait] if q not in questions_asked[chosen_trait]]
    if not available:
        # fallback: pick any question from that trait (allow repeats only if exhausted)
        available = QUESTION_BANK[chosen_trait][:]

    question = random.choice(available)
    return chosen_trait, question


def get_user_rating(prompt_text):
    """
    Ask user to input integer rating 1..5. Validate input.
    """
    while True:
        ans = input(prompt_text + " (1-5): ").strip()
        if ans.isdigit():
            val = int(ans)
            if 1 <= val <= 5:
                return val
        print("Invalid input — please enter an integer between 1 and 5.")


def normalize_scores():
    """
    For each trait, normalized_score = raw_score / (number_of_questions_for_trait)
    If no question asked for that trait, return 0.5 (neutral baseline).
    This yields values in [0,1].
    """
    normalized = {}
    for t in TRAITS:
        n = len(questions_asked[t])
        if n == 0:
            normalized[t] = 0.5
        else:
            normalized[t] = round(raw_scores[t] / n, 4)
    return normalized


def build_recommendation_prompt(qa_history, normalized_scores):
    """
    Build a human message that contains the Q&A history + normalized scores,
    and instructs the LLM to return EXACTLY a JSON object with 2-3 career recommendations.
    """
    # prepare readable Q&A list
    qa_lines = []
    for idx, (trait, question, rating) in enumerate(qa_history, start=1):
        qa_lines.append(f"{idx}. Trait: {trait} | Q: {question} | Rating: {rating}")

    human_text = (
        "You are an expert career counselor. Based on the user's answers below, "
        "recommend 2-3 career options. Return ONLY valid JSON (no extra text). "
        "Output schema must be:\n\n"
        "{\n"
        '  "recommendations": [\n'
        '    {"career": "<career name>", "reason": "<short reason (1-2 sentences)>"}\n'
        "  ]\n"
        "}\n\n"
        "Do not include any additional commentary or markdown — only the JSON.\n\n"
        "User Q&A (ordered):\n"
        + "\n".join(qa_lines)
        + "\n\n"
        "Final normalized RIASEC scores (each between 0 and 1):\n"
        + json.dumps(normalized_scores, indent=2)
        + "\n\n"
        "Give 2-3 career recommendations that match the user's profile. Keep reasons concise and tie each reason to the user's RIASEC strengths shown above."
    )

    return human_text


def ask_llm_for_recommendations(qa_history, normalized_scores):
    """
    Call Gemini with full QA history and normalized scores requesting JSON output.
    Returns parsed JSON (dict) or raw text if JSON parse fails.
    """
    human_prompt = build_recommendation_prompt(qa_history, normalized_scores)

    # append to chat history for context/history
    chat_msgs = chat_history_messages + [HumanMessage(content=human_prompt)]

    print("\nContacting LLM for career recommendations (JSON)...\n")
    # call LLM
    response = llm.invoke(chat_msgs)

    # the returned object may have .content attribute
    text = getattr(response, "content", None)
    if text is None:
        # try indexing into generations
        try:
            text = response.generations[0][0].text
        except Exception:
            text = str(response)

    # sanitize and parse JSON
    text_str = text.strip()
    # Try to extract JSON substring if LLM accidentally included text
    json_obj = None
    try:
        json_obj = json.loads(text_str)
        return json_obj, text_str
    except Exception:
        # attempt to find first '{' to last '}' and parse
        start = text_str.find("{")
        end = text_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text_str[start:end+1]
            try:
                json_obj = json.loads(snippet)
                return json_obj, text_str
            except Exception:
                pass
    # if parsing fails, return None and raw text for debugging
    return None, text_str

def generate_llm_mcq_questions(qa_history, num_questions=NUM_LLM_GENERATED_QUESTIONS):
    """
    Uses Gemini to generate MCQ questions after the initial RIASEC stage.
    Produces `num_questions` MCQs, each with 4 answer options (A/B/C/D).
    Returns a list of dicts:
    [
        {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
        },
        ...
    ]
    """

    # Format QA history for LLM context
    qa_lines = []
    for idx, (trait, question, rating) in enumerate(qa_history, start=1):
        qa_lines.append(f"{idx}. Trait={trait} | Q=\"{question}\" | Rating={rating}")

    human_text = (
        "You are an expert in psychometric assessment and adaptive career interest testing.\n"
        "Based on the user's previous answers listed below, generate "
        f"{num_questions} multiple-choice questions (MCQ) that capture deeper information "
        "about the user's career interests. These MCQs should refine and maximize information "
        "gain for predicting career fit.\n\n"

        "### Rules for output:\n"
        f"- Generate exactly {num_questions} questions.\n"
        "- Each question must have exactly 4 options labeled A, B, C, D.\n"
        "- Options must be meaningful and distinct.\n"
        "- Output JSON ONLY in this exact schema:\n\n"
        "{\n"
        '  "questions": [\n'
        '    {\n'
        '      "question": "<text>",\n'
        '      "options": {\n'
        '        "A": "<text>",\n'
        '        "B": "<text>",\n'
        '        "C": "<text>",\n'
        '        "D": "<text>"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "### User Q&A history:\n"
        + "\n".join(qa_lines)
        + "\n\n"
        "Do NOT add explanations or commentary. Output JSON only."
    )

    chat_msgs = chat_history_messages + [HumanMessage(content=human_text)]
    print("\nGenerating MCQ questions using LLM...\n")
    response = llm.invoke(chat_msgs)

    text = getattr(response, "content", "").strip()

    # Try to parse JSON
    try:
        data = json.loads(text)
        return data.get("questions", [])
    except:
        # Fallback: try substring JSON extraction
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
    Asks the user the generated MCQs in the terminal.
    Updates qa_history with entries:
      ("MCQ", question, chosen_option)
    """
    print("\n=== LLM-Generated MCQ Stage ===")
    print(f"You will now answer {len(mcq_list)} additional questions.\n")

    for i, qdata in enumerate(mcq_list, start=1):
        print(f"\nMCQ {i}: {qdata['question']}")
        for opt_key, opt_text in qdata["options"].items():
            print(f"  {opt_key}. {opt_text}")

        # Input validation
        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ["A", "B", "C", "D"]:
                break
            print("Invalid choice — select A, B, C, or D.")

        qa_history.append(("MCQ", qdata["question"], choice))

        # Add to LLM context (optional)
        chat_history_messages.append(
            HumanMessage(content=f"User answered MCQ: Q='{qdata['question']}' | Choice={choice}")
        )


def main():
    print("=== Adaptive RIASEC Test Chatbot ===")
    print(f"You will be asked {NUM_QUESTIONS_TO_ASK} adaptive questions.")
    print("Rate each question from 1 (low/negative) to 5 (high/positive).")
    input("Press Enter to start...")

    asked = 0
    while asked < NUM_QUESTIONS_TO_ASK:
        trait, question = pick_next_question()
        print(f"\nQuestion {asked+1}: ({trait}) {question}")
        rating = get_user_rating("Your rating")

        # update bookkeeping
        contribution = SCORE_MAP.get(rating, 0.5)
        raw_scores[trait] += contribution
        questions_asked[trait].append(question)
        qa_history.append((trait, question, rating))

        # add to LLM conversation history for context (optional)
        chat_history_messages.append(HumanMessage(content=f"User answered: Trait={trait} | Q='{question}' | Rating={rating}"))

        asked += 1

    # Stage 1 Complete — 6 RIASEC questions
    print("\n--- Stage 1 (RIASEC) Complete ---")

    # Generate MCQ questions using LLM
    mcq_questions = generate_llm_mcq_questions(qa_history, NUM_LLM_GENERATED_QUESTIONS)

    # Ask MCQs if successfully generated
    if mcq_questions:
        run_mcq_stage(mcq_questions, qa_history)
    else:
        print("Skipping MCQ stage due to generation error.")

    # Now compute normalized RIASEC results AFTER MCQ stage (MCQs do not affect scores)
    print("\n--- All Questions Completed ---")
    normalized = normalize_scores()

    print("\nNormalized RIASEC scores (0-1):")
    for t in TRAITS:
        print(f" {t}: {normalized[t]}")

    print("\nNormalized RIASEC scores (0-1):")
    for t in TRAITS:
        print(f" {t}: {normalized[t]}")

    # prepare call to LLM for recommendations
    json_result, raw_text = ask_llm_for_recommendations(qa_history, normalized)

    print("\nLLM returned (raw):\n")
    print(raw_text)
    print("\nParsed JSON (if parse succeeded):\n")
    if json_result is not None:
        print(json.dumps(json_result, indent=2, ensure_ascii=False))
    else:
        print("Failed to parse JSON. Raw LLM text shown above.")

    # also show simplified object you might want to use in a web app
    # final normalized scores already 0-1 (per trait)
    print("\nYou can use the normalized RIASEC scores above to plot a chart (each value 0-1).")

    print("\nDone. Thank you!")


if __name__ == "__main__":
    main()
