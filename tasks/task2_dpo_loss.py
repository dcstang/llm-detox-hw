"""Task 2 — DPO loss [15 points].

Given log-probabilities of the chosen and rejected completion under
both the policy and a frozen reference, return:

    losses           — per-example shape (batch,)
    chosen_rewards   — beta * (policy_chosen - reference_chosen), detached
    rejected_rewards — beta * (policy_rejected - reference_rejected), detached

The DPO loss is:

    -log sigmoid( beta * (
        log pi(y+|x) - log pi_ref(y+|x)
      - log pi(y-|x) + log pi_ref(y-|x)
    ))

The chosen/rejected rewards do NOT feed the optimiser — they're a
logging signal: their margin should rise during training, and either
drifting strongly negative is a known DPO-collapse leading indicator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss for one batch of pairs.

    For each pair (chosen y_+, rejected y_-) on prompt x, compute

        L = -log sigmoid( beta * [
            log( pi(y_+|x) / pi_ref(y_+|x) )
          - log( pi(y_-|x) / pi_ref(y_-|x) )
        ] )

    where ``pi`` is the trainable policy and ``pi_ref`` is the frozen
    reference (the SFT-merged checkpoint in this homework). All four
    log-prob tensors are *per-row sums* of token log-probs over each
    completion — the caller aggregates per-row before calling this.

    Args:
        policy_chosen_logps: shape ``(batch,)`` — pi log-probs on chosen.
        policy_rejected_logps: shape ``(batch,)`` — pi log-probs on rejected.
        reference_chosen_logps: shape ``(batch,)`` — pi_ref log-probs on chosen.
        reference_rejected_logps: shape ``(batch,)`` — pi_ref log-probs on rejected.
        beta: KL anchoring strength (higher = stay closer to the reference).

    Returns:
        Tuple ``(losses, chosen_rewards, rejected_rewards)``:
        - ``losses``: shape ``(batch,)`` — per-row DPO loss. The caller
          takes ``.mean()`` to get the scalar batch loss.
        - ``chosen_rewards``: shape ``(batch,)`` —
          ``beta * (policy_chosen_logps - reference_chosen_logps).detach()``.
          Used for logging only — does NOT feed the optimiser.
        - ``rejected_rewards``: shape ``(batch,)`` — same on the rejected side.
    """
    # <YOUR CODE HERE>
    raise NotImplementedError("Task 2: implement dpo_loss")
