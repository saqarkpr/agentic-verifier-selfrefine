"""Prompt construction for the feedback-protocol ablation.

Split out of feedback_protocol_ablation.py on purpose. That script imports
torch, transformers and peft at module level, so nothing inside it could be
exercised without a GPU stack -- and the prompts are where the experiment can
go wrong in ways a GPU would never reveal:

  - a feedback string that leaks the true answer makes every gain meaningless;
  - a `retry` prompt that differs from the turn-1 prompt is not a control;
  - telling the model its answer was "incorrect" when the verifier accepted it
    measures the lie rather than the model (this was a real bug in an earlier
    version of the file, caught before it cost 45 minutes of GPU time).

All three are cheap to assert and impossible to notice by reading output, so
they live here as pure functions with tests/test_experiment_validity.py
pinning them.
"""
import re

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."
ANSWER_FORMAT = "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."

ALL_CONDITIONS = ["retry", "generic", "diagnostic", "actionable", "constrained"]


def base_prompt(equation_str):
    return f"Solve for x: {equation_str}\n{ANSWER_FORMAT}"


def _procedure_for(reason):
    """A concrete next action implied by the verifier's reason code.

    Deliberately procedural and never revealing the true answer -- these
    strings tell the model what to DO, not what the answer is. Leaking the
    solution would make the experiment meaningless.
    """
    return {
        "extraneous_root": "Check which values make the denominator zero and exclude them.",
        "incomplete": "Check whether the equation has another root you have not listed.",
        "does_not_satisfy": "Substitute each value back into the original equation and recompute.",
        "wrong": "Substitute each value back into the original equation and recompute.",
        "unparsable": "Write the final line exactly in the requested form and nothing else.",
    }.get(reason, "Substitute each value back into the original equation and recompute.")


def build_prompt(condition, equation_str, prev_candidate, result):
    """Turn-2/3 prompt for each condition. Turn 1 is always base_prompt.

    If the previous answer was CORRECT (only reachable with
    MEASURE_REGRESSION on), every condition falls back to a truthful
    confirmation prompt. Telling the model an answer was wrong when the
    verifier said it was right would poison the measurement.

    Every variant keeps the first line and the final format line identical to
    the training prompt; the conditions differ only in what is inserted
    between them, so the ladder isolates information content rather than
    prompt shape -- except `constrained`, which deliberately adds a second
    format statement and is the one condition meant to change shape.
    """
    if condition == "retry":
        return base_prompt(equation_str)

    if condition not in ALL_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    if result is not None and result.correct:
        # Truthful: the verifier accepted it. Do not claim otherwise.
        return (f"Solve for x: {equation_str}\n"
                f"A previous attempt answered {prev_candidate}, which the checker accepted. "
                f"Confirm or correct it.\n{ANSWER_FORMAT}")

    if condition == "generic":
        middle = "Your previous answer was incorrect."
    elif condition == "diagnostic":
        middle = f"Your previous answer {prev_candidate} was incorrect. {result.feedback}"
    else:  # actionable, constrained
        middle = (f"Your previous answer {prev_candidate} was incorrect. {result.feedback} "
                  f"{_procedure_for(result.reason)}")

    if condition == "constrained":
        return (f"Solve for x: {equation_str}\n"
                f"{middle}\n"
                f"Reply with brief work, then the final line and nothing after it.\n"
                f"{ANSWER_FORMAT}")
    return f"Solve for x: {equation_str}\n{middle}\n{ANSWER_FORMAT}"


def named_values(feedback_text):
    """Values the verifier explicitly called out, e.g. an extraneous root.

    Used for the feedback-utilisation metric: if the verifier named a bad
    value and the model's next answer no longer contains it, the model
    demonstrably acted on the message rather than just re-sampling.
    """
    return set(re.findall(r"-?\d+(?:/\d+)?", feedback_text or ""))
