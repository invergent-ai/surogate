"""Import all model modules so they register with Base.metadata."""

from surogate.core.db.models.platform import *  # noqa: F401,F403
from surogate.core.db.models.operate import *  # noqa: F401,F403
from surogate.core.db.models.compute import *  # noqa: F401,F403
from surogate.core.db.models.observe import *  # noqa: F401,F403
from surogate.core.db.models.evaluate import *  # noqa: F401,F403
from surogate.core.db.models.train import *  # noqa: F401,F403
from surogate.core.db.models.settings import *  # noqa: F401,F403
from surogate.core.db.models.trace import *  # noqa: F401,F403
