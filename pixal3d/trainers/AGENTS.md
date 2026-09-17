# TRAINERS

## OVERVIEW

Distributed training lifecycle, checkpointing, optimization, flow-matching objectives, and VAE trainers.

## WHERE TO LOOK

- `basic.py`: distributed setup, dataloaders, optimizer/scheduler, checkpoint save/load, logging, and training loop.
- `flow_matching/`: generic, sparse, and image-conditioned flow objectives.
- `flow_matching/mixins/`: composable classifier-free, text, image, and projected-image conditioning.
- `vae/`: shape and PBR decoder training.
- `utils.py`, `../utils/data_utils.py`, and `../utils/dist_utils.py`: shared training state and distributed helpers.

## CONVENTIONS

- Trainer classes are selected dynamically from JSON config names in `train.py`; preserve constructor/config compatibility.
- Checkpoints include distributed/data-sampler state. Changes to `ResumableSampler`, rank partitioning, or initialization order affect restart semantics.
- Flow-matching trainers compose mixins; keep shared optimization mechanics in the base trainer and condition-specific behavior in the appropriate mixin.
- Multi-node training requires consistent node rank, master address, port, and world-size settings.

## ANTI-PATTERNS

- Do not make checkpoint loading silently accept incompatible model or sampler state without an explicit compatibility decision.
- Do not pass condition-specific kwargs through a model forward signature that does not accept them.
- Avoid wildcard-import changes in `basic.py` without checking symbol collisions and all trainer variants.
