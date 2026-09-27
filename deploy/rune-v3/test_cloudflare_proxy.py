#!/usr/bin/env python3
"""Exercise nginx client-IP trust and rate limits without a model or GPU."""
import http.client
import http.server
import json
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time


class Backend(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        data = json.dumps({'client': self.headers.get('X-Forwarded-For'),
                           'auth': self.headers.get('Authorization')}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST

    def log_message(self, *args):
        pass


def main():
    nginx = shutil.which('nginx')
    if not nginx or not shutil.which('openssl'):
        raise SystemExit('Install nginx with http_realip_module and openssl to run this test')
    renderer = Path(__file__).with_name('render_nginx.py')
    with tempfile.TemporaryDirectory(prefix='rune-cf-test-') as directory:
        root = Path(directory)
        cert, key = root / 'tls.crt', root / 'tls.key'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-days', '1', '-subj', '/CN=rune.test', '-addext', 'subjectAltName=IP:127.0.0.1,DNS:rune.test',
            '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
        context = ssl.create_default_context(cafile=str(cert))
        backend = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        try:
            for cloudflare in (True, False):
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    port = listener.getsockname()[1]
                config = root / 'nginx.conf'
                command = [sys.executable, str(renderer), '--hostname', 'rune.test',
                    '--listen', f'127.0.0.1:{port}', '--certificate', str(cert), '--private-key', str(key),
                    '--prefix', str(root), '--output', str(config)]
                if cloudflare:
                    trusted = root / 'trusted.txt'
                    trusted.write_text('127.0.0.2/32\n')
                    command += ['--cloudflare-ips', str(trusted)]
                subprocess.run(command, check=True, capture_output=True)
                config.write_text(config.read_text().replace('server 127.0.0.1:8460;',
                    f'server 127.0.0.1:{backend.server_port};'))
                with (root / 'nginx-stderr.log').open('w') as log:
                    process = subprocess.Popen([nginx, '-p', str(root) + '/', '-c', str(config),
                        '-g', 'daemon off;'], stdout=subprocess.DEVNULL, stderr=log)
                    try:
                        deadline = time.monotonic() + 10
                        while True:
                            assert process.poll() is None, (root / 'nginx-stderr.log').read_text()
                            try:
                                with socket.create_connection(('127.0.0.1', port), timeout=0.1):
                                    break
                            except OSError:
                                assert time.monotonic() < deadline, 'nginx startup timed out'
                                time.sleep(0.02)

                        def request(peer, client, *, forwarded=None, host='rune.test', path='/v1/decisions'):
                            connection = http.client.HTTPSConnection('127.0.0.1', port, timeout=5,
                                source_address=(peer, 0), context=context)
                            try:
                                connection.request('POST', path, body=b'{}', headers={
                                    'Host': host, 'CF-Connecting-IP': client,
                                    'X-Forwarded-For': forwarded or client,
                                    'Authorization': 'Bearer test-only'})
                                response = connection.getresponse()
                                return response.status, dict(response.getheaders()), response.read()
                            finally:
                                connection.close()

                        if cloudflare:
                            status, headers, body = request('127.0.0.2', '198.51.100.1')
                            assert status == 200 and headers['Cache-Control'] == 'no-store'
                            assert json.loads(body) == {'client': '198.51.100.1', 'auth': 'Bearer test-only'}
                            status, headers, _ = request('127.0.0.2', '198.51.100.1', forwarded='203.0.113.99')
                            assert status == 429 and headers['Retry-After'] == '1'
                            assert headers['Cache-Control'] == 'no-store'
                            assert request('127.0.0.2', '198.51.100.2')[0] == 200
                            status, _, body = request('127.0.0.2', '2001:db8::1')
                            assert status == 200 and json.loads(body)['client'] == '2001:db8::1'
                            assert request('127.0.0.3', '198.51.100.3')[0] == 403
                            assert request('127.0.0.1', '198.51.100.4')[0] == 200
                            assert request('127.0.0.1', '198.51.100.5')[0] == 429
                            assert request('127.0.0.2', '198.51.100.6', host='other.test')[0] == 404
                            assert request('127.0.0.2', '198.51.100.7', path='/metrics')[0] == 404
                            print('Cloudflare: distinct clients, IPv6, 429, spoof rejection, host/routes and no-store passed')
                        else:
                            status, _, body = request('127.0.0.3', '198.51.100.1')
                            assert status == 200 and json.loads(body)['client'] == '127.0.0.3'
                            assert request('127.0.0.3', '198.51.100.2')[0] == 429
                            print('Direct mode: socket-peer limit still ignores forwarded client headers')
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        finally:
            backend.shutdown()
            backend.server_close()


if __name__ == '__main__':
    main()
