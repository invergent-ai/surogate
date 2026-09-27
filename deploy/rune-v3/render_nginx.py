#!/usr/bin/env python3
"""Render a local or launch-day TLS proxy config, without starting or reloading it."""
import argparse
from pathlib import Path
import re
import os
import pwd
import ipaddress

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hostname', required=True)
parser.add_argument('--worker-user', help='nginx worker user when installing as root')
parser.add_argument('--listen', default='127.0.0.1:8443')
parser.add_argument('--per-ip-rps', type=int, default=1,
                    help='requests/second per client IP (default: 1)')
parser.add_argument('--per-ip-burst', type=int, default=0,
                    help='extra immediate requests allowed per IP (default: 0)')
parser.add_argument('--cloudflare-ips', type=Path,
                    help='Cloudflare CIDR file: trust CF-Connecting-IP and restrict origin peers to these networks or loopback')
parser.add_argument('--certificate', type=Path, required=True)
parser.add_argument('--private-key', type=Path, required=True)
parser.add_argument('--prefix', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
if not 1 <= a.per_ip_rps <= 1000000:
    parser.error('--per-ip-rps must be between 1 and 1000000')
if not 0 <= a.per_ip_burst <= 65535:
    parser.error('--per-ip-burst must be between 0 and 65535')
if not re.fullmatch(r'[A-Za-z0-9.-]+', a.hostname):
    parser.error('hostname must be a DNS name')
if not re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}', a.listen):
    parser.error('listen must be an IPv4 address and port, e.g. 0.0.0.0:443')
values = {'HOSTNAME': a.hostname.lower(), 'LISTEN': a.listen, 'USER_DIRECTIVE': '',
          'TRUSTED_PROXY_CONFIG': '', 'ORIGIN_PEER_CHECK': '',
          'PER_IP_RPS': str(a.per_ip_rps),
          # nginx's zero-burst policy is expressed by omitting the burst parameter.
          'PER_IP_BURST': f' burst={a.per_ip_burst} nodelay' if a.per_ip_burst else ''}
if a.cloudflare_ips:
    try:
        networks = [ipaddress.ip_network(line.strip()) for line in a.cloudflare_ips.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
    except (OSError, ValueError) as error:
        parser.error(f'invalid Cloudflare CIDR file: {error}')
    if not networks or any(network.prefixlen == 0 for network in networks):
        parser.error('Cloudflare CIDRs must be nonempty and cannot trust all addresses')
    networks = list(dict.fromkeys(networks))
    lines = [f'set_real_ip_from {network};' for network in networks]
    lines += ['real_ip_header CF-Connecting-IP;',
              # Check the original socket peer after realip rewrites remote_addr.
              'geo $realip_remote_addr $rune_origin_peer {',
              '    default 0;', '    127.0.0.1/32 1;', '    ::1/128 1;']
    lines += [f'    {network} 1;' for network in networks
              if str(network) not in ('127.0.0.1/32', '::1/128')]
    lines += ['}']
    values['TRUSTED_PROXY_CONFIG'] = '\n    '.join(lines)
    values['ORIGIN_PEER_CHECK'] = 'if ($rune_origin_peer = 0) { return 403; }'
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
