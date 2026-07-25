FROM fugu/ale-ubuntu22-docker:20260721-dindfix-v2

# The public cost_optimization_1 software wrapper is pinned to this task-local
# interpreter path.  The parent image already contains the exact declared
# Python 3.10.12 + pandas 2.3.3 bytes.  Build a task-local venv and bind its
# site-packages to those pinned in-image bytes so PYTHONNOUSERSITE=1, which the
# public wrapper intentionally sets, still yields the declared environment.
USER root
RUN python3 -m venv --without-pip --system-site-packages \
        /media/user/data/agenthle/computing_math/cost_optimization_1/base/.runtime/python-3.10.12-pandas-2.3.3 \
    && rm -rf \
        /media/user/data/agenthle/computing_math/cost_optimization_1/base/.runtime/python-3.10.12-pandas-2.3.3/lib/python3.10/site-packages \
    && ln -s /home/user/.local/lib/python3.10/site-packages \
        /media/user/data/agenthle/computing_math/cost_optimization_1/base/.runtime/python-3.10.12-pandas-2.3.3/lib/python3.10/site-packages \
    && chown user:user \
        /media/user/data/agenthle/computing_math/cost_optimization_1/base \
    && chown user:user \
        /media/user/data/agenthle/computing_math
USER user
