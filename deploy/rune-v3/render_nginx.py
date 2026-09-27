#!/usr/bin/env python3
"""Render a local or launch-day TLS proxy config, without starting or reloading it."""
import argparse
from pathlib import Path
import re
import os
import pwd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hostname', required=True)
parser.add_argument('--worker-user', help='nginx worker user when installing as root')
parser.add_argument('--listen', default='127.0.0.1:8443')
parser.add_argument('--certificate', type=Path, required=True)
parser.add_argument('--private-key', type=Path, required=True)
parser.add_argument('--prefix', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9.-]+', a.hostname):
    parser.error('hostname must be a DNS name')
if not re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}', a.listen):
    parser.error('listen must be an IPv4 address and port, e.g. 0.0.0.0:443')
values = {'HOSTNAME': a.hostname, 'LISTEN': a.listen, 'USER_DIRECTIVE': ''}
worker = None
if a.worker_user:
    if os.geteuid() != 0: parser.error('--worker-user requires root to own the temp directories')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', a.worker_user): parser.error('invalid worker user')
    worker = pwd.getpwnam(a.worker_user)
    values['USER_DIRECTIVE'] = f'user {a.worker_user};'
for name, path in [('PREFIX', a.prefix), ('CERTIFICATE', a.certificate), ('PRIVATE_KEY', a.private_key)]:
    path = path.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(path)):
        parser.error('config paths must contain only letters, numbers, /, _, . and -')
    values[name] = str(path)
for path in (a.certificate, a.private_key):
    if not path.is_file(): parser.error(f'missing TLS file: {path}')
for name in ('body', 'proxy', 'fastcgi', 'scgi', 'uwsgi'):
    (a.prefix / name).mkdir(parents=True, exist_ok=True)
    if worker: os.chown(a.prefix / name, worker.pw_uid, worker.pw_gid)
text = Path(__file__).with_name('nginx.conf.in').read_text()
for name, value in values.items(): text = text.replace('@' + name + '@', value)
a.output.write_text(text)
