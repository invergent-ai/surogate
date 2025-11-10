from pathlib import Path

import evalscope
from evalscope.constants import EvalType
from evalscope.run import run_single_task, run_task

from surogate.utils.config import load_config


class SurogateEval:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])
        self.task_config = evalscope.TaskConfig(
            model=self.config['target']['model'],
            api_url=self.config['target']['api_url'],
            eval_type=EvalType.SERVICE,
        )

    def run(self):
        run_task(task_cfg=self.task_config)


