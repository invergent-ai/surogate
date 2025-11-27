git clone -b v2.8.3 --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=20 uv run python setup.py install
