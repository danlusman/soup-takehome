"""Single source of truth for the task: prompt wording, target string, and scoring.

prepare_data.py (training targets) and eval_task.py / verify.py (scoring) both import
this module, so the prompt the model is trained on is byte-identical to the prompt it is
evaluated on. If you change anything here, regenerate the data AND re-run the baseline.

Task: 3-class sentiment of a Russian product review (RuReviews), answered as strict JSON
    {"тональность": "<негативный|нейтральный|позитивный>"}
"""
from __future__ import annotations

import json
import re

JSON_KEY = "тональность"
# source label id -> Russian label word (ai-forever/ru-reviews-classification: 0 neg, 1 neu, 2 pos;
# verified from its `label_text` column, see DECISIONS.md D-03)
LABELS = {0: "негативный", 1: "нейтральный", 2: "позитивный"}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}
SOURCE_LABEL_TEXT = {0: "negative", 1: "neutral", 2: "positive"}

PROMPT_TEMPLATE = (
    "Определи тональность отзыва покупателя о товаре.\n"
    "Ответь строго одной строкой JSON без пояснений и без markdown: "
    '{{"тональность": "<метка>"}}, где <метка> — одно из: '
    '"негативный", "нейтральный", "позитивный".\n\n'
    "Отзыв:\n{review}"
)


def build_prompt(review: str) -> str:
    return PROMPT_TEMPLATE.format(review=review)


def build_target(label_id: int) -> str:
    return json.dumps({JSON_KEY: LABELS[label_id]}, ensure_ascii=False)


def build_messages(review: str, label_id: int | None = None) -> list[dict]:
    """No system message on purpose: Qwen2.5's chat template then inserts its default
    system prompt. Training and eval must both go through the same apply_chat_template."""
    msgs = [{"role": "user", "content": build_prompt(review)}]
    if label_id is not None:
        msgs.append({"role": "assistant", "content": build_target(label_id)})
    return msgs


_STEMS = {0: "негатив", 1: "нейтрал", 2: "позитив"}


def parse_response(text: str) -> dict:
    """Score one model output.

    strict_valid : the whole stripped output is a JSON object with exactly one key
                   JSON_KEY whose value is one of the 3 label words. Nothing else.
    strict_label : label id if strict_valid else None
    lenient_label: label id if exactly one label stem occurs anywhere (case-insensitive),
                   else None. Separates "knows the answer" from "obeys the format".
    """
    out = {"strict_valid": False, "strict_label": None, "lenient_label": None,
           "json_parse_ok": False}
    s = text.strip()
    try:
        obj = json.loads(s)
        out["json_parse_ok"] = True
        if (isinstance(obj, dict) and list(obj.keys()) == [JSON_KEY]
                and obj[JSON_KEY] in LABEL_TO_ID):
            out["strict_valid"] = True
            out["strict_label"] = LABEL_TO_ID[obj[JSON_KEY]]
    except (json.JSONDecodeError, TypeError):
        pass
    low = s.lower()
    found = [lid for lid, stem in _STEMS.items() if re.search(stem, low)]
    if len(found) == 1:
        out["lenient_label"] = found[0]
    return out
