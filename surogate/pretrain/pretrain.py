from typing import List, Optional, Union

from swift.utils import get_logger

import pyllmq

gpus = pyllmq.get_num_gpus()
print(gpus)