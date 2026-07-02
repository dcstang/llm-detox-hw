"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

from typing import Sequence


def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion. Higher = better.

    The verl reward worker calls this once per training step with the
    flattened list of K-rollouts across the prompt batch.

    Args:
        texts: completions to score, one entry per completion.
        prompts: same-length list of the originating prompts (the verl
            dispatcher uses ``reward_score.prompt_conditioned`` below
            to decide whether to pass these). Set the attribute to
            ``False`` if your design is purely response-side.

    Returns:
        ``list[float]`` of the same length as ``texts``. Higher = better.

    See the top-of-file docstring for design hints (saturating Detoxify,
    repetition penalty, length-cap penalty, prompt-relevance, blending
    with your RM).
    """
    from detoxify import Detoxify
    import math

    _detox = Detoxify("original")
    raw = _detox.predict(list(texts))
    tox_scores = raw["toxicity"] if isinstance(raw["toxicity"], list) else [raw["toxicity"]]

    rewards = []
    for i, (text, tox) in enumerate(zip(texts, tox_scores)):
        # Saturating detox: benign floor — once clearly non-toxic, no extra incentive
        detox_reward = min(1.0 - float(tox), 0.8)

        # Trigram repetition penalty
        words = text.lower().split()
        if len(words) >= 3:
            trigrams = [tuple(words[j:j+3]) for j in range(len(words) - 2)]
            unique_ratio = len(set(trigrams)) / len(trigrams)
        else:
            unique_ratio = 1.0
        rep_penalty = 1.0 - unique_ratio  # 0 = no repeats, 1 = all repeats

        # Prompt-relevance: bag-of-words overlap with prompt (if available)
        if prompts is not None:
            prompt_words = set(prompts[i].lower().split())
            response_words = set(words)
            if prompt_words:
                overlap = len(prompt_words & response_words) / len(prompt_words)
                # Penalise pure echo (overlap > 0.8) and reward moderate relevance
                if overlap > 0.8:
                    relevance = 0.3  # penalise trivial echoing
                else:
                    relevance = min(overlap * 2, 0.5)
            else:
                relevance = 0.0
        else:
            relevance = 0.0

        reward = detox_reward - 0.3 * rep_penalty + 0.2 * relevance
        rewards.append(float(reward))

    return rewards


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
