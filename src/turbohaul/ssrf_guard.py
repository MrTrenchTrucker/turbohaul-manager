"""SSRF guard for pull-url + pull-hf per v0.2 ARCHITECTURE.md §9.1.

Per security finding F3 CRITICAL must-fix. Defends against:
  - http/file/ftp/gopher/dict/ldap/data scheme abuse → https-only
  - Internal-network endpoints (e.g. IMDS) → blocked via the deny ranges
  - NAT64 64:ff9b::/96 + IPv4-compat IPv6 ::/96 ( bypass classes)
  - DNS rebinding → resolve-once + peer-IP verify (caller responsibility)
  - HF_API_KEY exfil to non-HF hosts → host allowlist
"""
import ipaddress
import logging
import socket
from urllib.parse import urlparse


log = logging.getLogger(__name__)


class UrlSafetyError(ValueError):
    """URL failed safety validation (scheme, host, resolved IP)."""


ALLOWED_SCHEMES: set[str] = {"https"}


# Private + link-local + reserved IP ranges (IPv4 + IPv6)
DENY_IPV4_NETWORKS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),     # RFC1918 (includes private overlay ranges)
    ipaddress.IPv4Network("172.16.0.0/12"),  # RFC1918
    ipaddress.IPv4Network("192.168.0.0/16"), # RFC1918
    ipaddress.IPv4Network("127.0.0.0/8"),    # loopback
    ipaddress.IPv4Network("169.254.0.0/16"), # link-local (IMDS lives here)
    ipaddress.IPv4Network("100.64.0.0/10"),  # CGNAT
    ipaddress.IPv4Network("0.0.0.0/8"),      # "this network"
    ipaddress.IPv4Network("224.0.0.0/4"),    # multicast
    ipaddress.IPv4Network("240.0.0.0/4"),    # reserved
]

DENY_IPV6_NETWORKS: list[ipaddress.IPv6Network] = [
    ipaddress.IPv6Network("::1/128"),         # loopback
    ipaddress.IPv6Network("fc00::/7"),        # unique-local
    ipaddress.IPv6Network("fe80::/10"),       # link-local
    ipaddress.IPv6Network("ff00::/8"),        # multicast
    ipaddress.IPv6Network("64:ff9b::/96"),    # NAT64 ( bypass class)
    ipaddress.IPv6Network("::/96"),           # IPv4-compat IPv6
    ipaddress.IPv6Network("::ffff:0:0/96"),   # IPv4-mapped IPv6
    ipaddress.IPv6Network("2001:db8::/32"),   # documentation
]


def is_blocked_ip(ip_str: str) -> bool:
    """True if the IP falls in a denied network."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Unparseable = block

    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in DENY_IPV4_NETWORKS)
    if isinstance(ip, ipaddress.IPv6Address):
        return any(ip in net for net in DENY_IPV6_NETWORKS)
    return True


def resolve_safely(host: str) -> str:
    """Resolve hostname to a SINGLE IP and verify it's not in a denied range.

    Returns the resolved IP. Raises UrlSafetyError if host or IP fails checks.

    Caller MUST connect to this resolved IP (not re-resolve) to defeat DNS
    rebinding — the manager's pull worker passes resolved_ip explicitly.
    """
    try:
        # getaddrinfo returns all records; we pin to the first
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UrlSafetyError(f"DNS resolution failed for {host}: {e}") from e
    if not infos:
        raise UrlSafetyError(f"no DNS records for {host}")
    # Pick the first resolved address
    family, _socktype, _proto, _canonname, sockaddr = infos[0]
    ip = sockaddr[0]
    if is_blocked_ip(ip):
        raise UrlSafetyError(
            f"host {host} resolves to {ip} which is in a denied network "
            "(RFC1918 / link-local / NAT64 / IPv4-compat IPv6 — v0.2 §9.1)"
        )
    return ip


def validate_pull_url(url: str) -> tuple[str, str]:
    """Validate a pull URL. Returns (host, resolved_ip).

    Raises UrlSafetyError on scheme / host / IP failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UrlSafetyError(
            f"scheme {parsed.scheme!r} not in {ALLOWED_SCHEMES} (only https:// allowed)"
        )
    if not parsed.hostname:
        raise UrlSafetyError("URL missing hostname")

    # Detect IP literal in hostname (e.g., https://10.0.0.1/x). Must do the
    # ip_address parse + the deny check in TWO try blocks — UrlSafetyError
    # inherits ValueError, so a single try/except ValueError swallows my own raise.
    ip_literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip_literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        ip_literal = None

    if ip_literal is not None:
        if is_blocked_ip(str(ip_literal)):
            raise UrlSafetyError(
                f"URL host is IP literal {parsed.hostname} in denied range"
            )
        return parsed.hostname, str(ip_literal)

    resolved = resolve_safely(parsed.hostname)
    return parsed.hostname, resolved


def is_hf_host(host: str, allowlist: list[str]) -> bool:
    """Match host against HF allowlist (huggingface.co + subdomains, hf.co + subdomains)."""
    host_l = host.lower()
    for allowed in allowlist:
        a = allowed.lower()
        if host_l == a or host_l.endswith("." + a):
            return True
    return False
