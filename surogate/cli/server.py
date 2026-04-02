import asyncio
import os
from pathlib import Path
import sys
import argparse
import signal
from threading import Thread
import traceback
import urllib3

from surogate.core.config.loader import load_config
from surogate.core.config.server_config import ServerConfig
from surogate.utils.logger import get_logger

logger = get_logger()
urllib3.disable_warnings()

# The uvicorn server instance -- set by run_server(), used by callers
# that need to tell the server to exit (e.g. signal handlers).
_server = None

# Shutdown event -- used to wake the main loop on signal
_shutdown_event = None

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, nargs='?', default=None, help='Path or HTTP(s) URL to config file')

    return parser

def run_server(config: ServerConfig, frontend_path: Path = Path(__file__).resolve().parent.parent / "frontend" / "dist"):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to (auto-increments if in use)
        frontend_path: Path to frontend build directory (optional)
        silent: Suppress startup messages

    Note:
        Signal handlers are NOT registered here so that embedders
        (e.g. Colab notebooks) keep their own interrupt semantics.
        Standalone callers should register handlers after calling this.
    """
    global _server, _shutdown_event
    
    import uvicorn
    import time
    from surogate.server.app import app, setup_frontend
    from threading import Thread, Event
     
    # Make config available to routes via app.state
    app.state.config = config

    # Setup frontend if path provided
    if frontend_path:
        if setup_frontend(app, frontend_path):
            print(f"✅ Frontend loaded from {frontend_path}")
        else:
            print(f"⚠️ Frontend not found at {frontend_path}")
                
    # Start the server in a separate thread so that we can listen for signals
    config = uvicorn.Config(
        app, host=config.host, port=config.port, log_level="info", access_log = False, workers=config.workers
    )
    _server = uvicorn.Server(config)
    _shutdown_event = Event()
    
    def _run():
        asyncio.run(_server.serve())
    
    thread = Thread(target = _run, daemon = True)
    thread.start()
    time.sleep(3)
    
    # Expose a shutdown callable via app.state so the /api/shutdown endpoint
    # can trigger graceful shutdown without circular imports.
    def _trigger_shutdown():
        _graceful_shutdown(_server)
        if _shutdown_event is not None:
            _shutdown_event.set()

    app.state.trigger_shutdown = _trigger_shutdown
    
    return app
    
    
def _graceful_shutdown(server = None):
    """Explicitly shut down all subprocess backends and the uvicorn server.

    Called from signal handlers to ensure child processes are cleaned up
    before the parent exits. This is critical on Windows where atexit
    handlers are unreliable after Ctrl+C.
    """
    logger.info("Graceful shutdown initiated — cleaning up subprocesses...")

    # 1. Shut down uvicorn server (releases the listening socket).
    #    This triggers the lifespan teardown which calls cleanup_children().
    if server is not None:
        server.should_exit = True
    
    
if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])
    config_path = args.config
    
    if not config_path:
        default_config = Path.home() / ".surogate" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    if not config_path:
        print("Error: No config file provided and ~/.surogate/config.yaml not found.", file=sys.stderr)
        sys.exit(1)

    config = load_config(ServerConfig, config_path)

    try:
        run_server(config)
    except Exception:
        sys.stderr.write("\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("ERROR: Surogate Server failed to start.\n")
        sys.stderr.write("=" * 60 + "\n")
        traceback.print_exc(file = sys.stderr)
        sys.stderr.write("\n")
        sys.stderr.flush()
        sys.exit(1)

    # Signal handler -- ensures subprocess cleanup on Ctrl+C
    def _signal_handler(signum, frame):
        _graceful_shutdown(_server)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # On Windows, some terminals send SIGBREAK for Ctrl+C / Ctrl+Break
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    # Keep running until shutdown signal.
    # NOTE: Event.wait() without a timeout blocks at the C level on Linux,
    # which prevents Python from delivering SIGINT (Ctrl+C).  Using a
    # short timeout in a loop lets the interpreter process pending signals.
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout = 1)
