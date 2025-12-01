import json
import random

# Load RIASEC question bank
with open("questions.json", "r") as f:
    QUESTION_BANK = json.load(f)

# Track user data
riasec_scores = {k: 0 for k in "RIASEC"}
questions_asked = {k: [] for k in "RIASEC"}
total_questions_asked = 0
MAX_QUESTIONS = 6

def score_mapping(user_input):
    mapping = {
        1: 0.0,
        2: 0.25,
        3: 0.5,
        4: 0.75,
        5: 1.0
    }
    return mapping.get(user_input, 0.5)

def pick_next_question():
    # choose the trait with the fewest questions asked
    fewest_trait = min(questions_asked, key=lambda k: len(questions_asked[k]))

    # pick a random unused question
    remaining_questions = list(set(QUESTION_BANK[fewest_trait]) - set(questions_asked[fewest_trait]))

    if not remaining_questions:
        # fallback: choose ANY trait randomly
        all_traits = list("RIASEC")
        random_trait = random.choice(all_traits)
        question = random.choice(QUESTION_BANK[random_trait])
        return random_trait, question

    question = random.choice(remaining_questions)
    return fewest_trait, question

def ask_question():
    global total_questions_asked

    trait, question = pick_next_question()

    print(f"\nQuestion {total_questions_asked+1}: {question}")
    user_input = int(input("Rate (1-5): "))

    # scoring
    riasec_scores[trait] += score_mapping(user_input)

    # mark this question asked
    questions_asked[trait].append(question)
    total_questions_asked += 1

def normalize_scores():
    normalized = {}
    for trait in "RIASEC":
        count = len(questions_asked[trait])
        if count == 0:
            normalized[trait] = 0.5
        else:
            normalized[trait] = riasec_scores[trait] / count
    return normalized

if __name__ == "__main__":
    print("RIASEC Adaptive Test Started...\n")

    while total_questions_asked < MAX_QUESTIONS:
        ask_question()

    final_scores = normalize_scores()

    print("\nFinal Normalized RIASEC Scores (0-1):")
    print(final_scores)
