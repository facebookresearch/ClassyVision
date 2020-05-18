import logging
from typing import Any, Collection, Dict, Optional

from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.util import get_checkpoint_dict, save_checkpoint
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook

@register_hook("csv")
class CSVHook(ClassyHook):
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self) -> None:
        super().__init__()

    def on_step(self, task) -> None:
        with open('/tmp/predictions.txt', 'a+') as f:
            sample_ids = task.last_batch.sample["id"]
            scores = task.last_batch.output.argmax(dim=1).tolist()
            targets = task.last_batch.sample["target"].argmax(dim=1).tolist()
            for sample_id, score, target in zip(sample_ids, scores, targets):
                f.write(f'{sample_id}\t{score}\t{target}\n')

