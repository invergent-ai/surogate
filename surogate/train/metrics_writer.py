import json
import os
import time

DEFAULT_METRICS_PATH = '/tmp/surogate_metrics.jsonl'
ENV_METRICS_PATH = 'SUROGATE_METRICS_PATH'


class MetricsWriter:
    """Writes training metrics as JSONL lines to a file on disk.

    Usage:
        writer = MetricsWriter()
        for step, batch in enumerate(train_loader):
            loss = model(batch)
            writer.track(step, epoch=epoch, loss=loss.item(), lr=scheduler.get_last_lr()[0])
        writer.close()
    """

    def __init__(self, output_path=None):
        if output_path is None:
            output_path = os.environ.get(ENV_METRICS_PATH, DEFAULT_METRICS_PATH)
        self._path = output_path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._file = open(self._path, 'a')

    def track(self, step, epoch=None, **metrics):
        entry = {'step': step, 'ts': time.time()}
        if epoch is not None:
            entry['epoch'] = epoch
        entry.update(metrics)
        self._file.write(json.dumps(entry) + '\n')
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
