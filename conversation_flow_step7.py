from typing import Dict, Any

from rag_with_llm_step6 import analyze_patient


def zone_label(zone: str) -> str:
    labels = {
        "Red": "Zone: 🔴 Red – उच्च धोक्याची पातळी",
        "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
        "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
    }
    return labels.get(zone, f"Zone: {zone}")


def pretty_print_result(step_label: str, result: Dict[str, Any]):
    """Print results in a patient-friendly way."""
    llm = result["llm_result"]
    zone = llm["final_zone"]
    print(f"\n========== {step_label} ==========")
    print("Patient text:")
    print(result["input_text"])
    print()
    print(zone_label(zone))
    print(llm["patient_symptoms_line"])
    print("काय करावे:")
    print(llm["patient_action_line"])
    print("\nनोट: हा केवळ प्राथमिक अंदाज आहे, कृपया डॉक्टरांचा सल्ला नक्की घ्या.")
    print("\nFollow-up question from model:")
    fq = llm["followup_question"]
    print(fq if fq else "(none)")
    print("====================================\n")


def run_conversation():
    """
    Multi-turn conversation with these rules:

    - Gemini ALONE decides whether to ask a follow-up question.
    - We NEVER invent our own default questions.
    - We ONLY limit how many follow-ups can happen:

        * If zone == Red:
            - max_followups = 3
        * If zone == Yellow or Orange:
            - max_followups = 5

    - Flow:
        1) Ask initial complaint
        2) Run analyze_patient()
        3) If Gemini returns followup_question (non-empty) AND we are under max_followups:
               ask user and append answer, then go to next round
           Else:
               stop and show final decision
    """

    print("\n=== AI Symptom Checker: Multi-turn Demo (Step 7) ===")
    print("Please type your main complaint in Marathi.\n")

    text = input("Patient complaint: ").strip()
    if not text:
        print("No input given. Exiting.")
        return

    round_num = 1
    followup_count = 0
    max_followups = 0
    zone = None

    while True:
        result = analyze_patient(text)
        llm = result["llm_result"]

        if round_num == 1:
            # Decide zone and max follow-ups after first result
            zone = llm["final_zone"]
            if zone == "Red":
                max_followups = 3
            else:
                max_followups = 5

        pretty_print_result(f"STEP {round_num} RESULT", result)

        model_q = (llm["followup_question"] or "").strip()

        # If model is not asking anything more -> stop
        if not model_q:
            print("No more follow-up questions. Final triage decision above.\n")
            break

        # If we already hit max follow-ups for this zone -> stop
        if followup_count >= max_followups:
            print("Maximum follow-up questions reached for this zone.")
            print("Please act according to the advice above.\n")
            break

        # Otherwise, ask the model's follow-up question
        print("Please answer the follow-up question (in Marathi):")
        print(model_q)
        answer = input("Your answer: ").strip()

        # Append answer to text for the next round
        text = (
            text
            + "\n\n"
            + f"अतिरिक्त माहिती (फॉलो-अप उत्तर {followup_count + 1}): "
            + answer
        )

        followup_count += 1
        round_num += 1

    print("Conversation finished.\n")


if __name__ == "__main__":
    run_conversation()

