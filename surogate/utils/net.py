import socket
from typing import Optional


def find_free_port(start_port: Optional[int] = None, end_port: int = 100) -> int:
    if start_port is None:
        start_port = 0
    for port in range(start_port, start_port + end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('', port))
                port = sock.getsockname()[1]
                break
            except OSError:
                pass
    return port

def find_node_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    return s.getsockname()[0]
