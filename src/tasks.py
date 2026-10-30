"""Per-task prompt construction and grading. Grading is a deterministic function of the text."""
import re

GSM8K_SUFFIX = "\nPlease reason step by step, and put your final answer after '####'."

MMLU_PRO_TEMPLATE = (
    "The following is a multiple choice question about {subject}. Think step by step and then "
    'finish your answer with "The answer is (X)" where X is the correct letter choice.\n\n'
    "Question: {question}\nOptions:\n{options}"
)
LETTERS = "ABCDEFGHIJ"


def mmlu_pro_prompt(question, options, subject):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return MMLU_PRO_TEMPLATE.format(subject=subject, question=question, options=opts)


def user_content(rec):
    if rec.get("task", "gsm8k") == "gsm8k":
        return rec["question"] + GSM8K_SUFFIX
    return rec["prompt"]


def extract_numeric(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    return s.replace(",", "").rstrip(".")


def extract_letter(text):
    # MMLU-Pro reference extraction order: "answer is (X)", then "Answer: X", then last letter.
    m = re.findall(r"answer is \(?([A-J])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"[aA]nswer:\s*\(?([A-J])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m[-1] if m else None


def answer(text, rec):
    return extract_numeric(text) if rec.get("task", "gsm8k") == "gsm8k" else extract_letter(text)


def grade(text, rec):
    return answer(text, rec) == rec["gold"]
