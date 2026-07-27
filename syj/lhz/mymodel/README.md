# mymodel_reward_v2

This folder is an experimental copy of `syj/lhz/mymodel`. The original
`mymodel` folder is not modified.

## Main Changes

- Default reward mode is `tail_cost_curiosity`.
- The reward keeps sample-level set F1 shaping, but shifts the main terminal
  signal toward weighted tail-aware F-score.
- Correct rare-label actions receive an extra bonus.
- False positives use dynamic asymmetric penalties: early training is looser,
  later training becomes stricter.
- After supervised pretraining, the trainer records labels that the pretrained
  model ranks low or predicts with low confidence. RL receives curiosity bonus
  only when it correctly selects those true labels.
- Prioritized replay gives extra priority to rare and pretrained-missed true
  labels.
- Evaluation still reports the original `macro_f1`, and also reports
  `supported_macro_f1`, which excludes labels absent from the evaluation split.

## Suggested Run

```bash
python3 syj/lhz/mymodel_reward_v2/main.py -episode 100
```

For a quick smoke test:

```bash
python3 syj/lhz/mymodel_reward_v2/main.py -episode 1 -pretrain_epochs 1 -mem_capacity 1000
```

## Reward Components

`tail_cost_curiosity` uses:

```text
R = potential(weighted F_beta)
  + rare true-positive bonus
  + pretrained-uncertainty true-positive bonus
  - dynamic false-positive cost
  - rare false-negative terminal cost
  - cardinality mismatch cost
```

The curiosity term is deliberately gated by correctness. It does not reward
novel or rare false positives.
