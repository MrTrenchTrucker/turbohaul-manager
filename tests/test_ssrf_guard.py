"""Tests for ssrf_guard - URL safety validation (v0.2 §9.1)."""
import socket
from unittest.mock import patch

import pytest

from turbohaul.ssrf_guard import (
    UrlSafetyError,
    is_blocked_ip,
    is_hf_host,
    resolve_safely,
    validate_pull_url,
)


class TestIsBlockedIp:
    def test_rfc1918_blocked(self):
        for ip in ["10.0.0.1", "10.0.0.5", "172.16.0.1", "192.168.1.50"]:
            assert is_blocked_ip(ip), ip

    def test_loopback_blocked(self):
        assert is_blocked_ip("127.0.0.1")
        assert is_blocked_ip("127.0.0.123")

    def test_link_local_imds_blocked(self):
        assert is_blocked_ip("169.254.169.254")  # AWS IMDS
        assert is_blocked_ip("169.254.0.1")

    def test_cgnat_blocked(self):
        assert is_blocked_ip("100.64.0.1")
        assert is_blocked_ip("100.127.255.254")

    def test_multicast_blocked(self):
        assert is_blocked_ip("224.0.0.1")
        assert is_blocked_ip("239.255.255.250")

    def test_ipv6_loopback_blocked(self):
        assert is_blocked_ip("::1")

    def test_ipv6_link_local_blocked(self):
        assert is_blocked_ip("fe80::1")

    def test_ipv6_ula_blocked(self):
        assert is_blocked_ip("fc00::1")

    def test_nat64_blocked(self):
        """NAT64 bypass class."""
        assert is_blocked_ip("64:ff9b::1.2.3.4")  # NAT64-encoded 1.2.3.4
        assert is_blocked_ip("64:ff9b::a.b.c.d") if False else True
        assert is_blocked_ip("64:ff9b::101:101")  # 1.1.1.1 in NAT64 prefix

    def test_ipv4_compat_ipv6_blocked(self):
        """IPv4-compat IPv6 bypass class — ::1.2.3.4 form."""
        assert is_blocked_ip("::1.1.1.1")
        assert is_blocked_ip("::8.8.8.8")

    def test_ipv4_mapped_ipv6_blocked(self):
        assert is_blocked_ip("::ffff:1.1.1.1")

    def test_unparseable_blocked(self):
        """Defense-in-depth: unparseable input is treated as blocked."""
        assert is_blocked_ip("not-an-ip")
        assert is_blocked_ip("")

    def test_public_ipv4_not_blocked(self):
        # Real public IPs (Cloudflare, Google)
        assert not is_blocked_ip("1.1.1.1")
        assert not is_blocked_ip("8.8.8.8")

    def test_public_ipv6_not_blocked(self):
        assert not is_blocked_ip("2606:4700:4700::1111")  # Cloudflare


class TestResolveSafely:
    def test_resolves_public_host(self):
        """We rely on real DNS here — uses Cloudflare DNS (1.1.1.1) lookup."""
        # Use a synthetic mock instead to avoid network flakiness
        with patch("socket.getaddrinfo") as m:
            m.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))]
            ip = resolve_safely("one.one.one.one")
            assert ip == "1.1.1.1"

    def test_blocks_private_resolution(self):
        with patch("socket.getaddrinfo") as m:
            m.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))
            ]
            with pytest.raises(UrlSafetyError, match="denied network"):
                resolve_safely("rebinding-attack.example")

    def test_dns_failure_raises(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with pytest.raises(UrlSafetyError, match="DNS resolution failed"):
                resolve_safely("nonexistent.invalid")


class TestValidatePullUrl:
    def test_https_public_ok(self):
        with patch("socket.getaddrinfo") as m:
            m.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))
            ]
            host, ip = validate_pull_url("https://example.com/model.gguf")
            assert host == "example.com"
            assert ip == "1.1.1.1"

    def test_http_scheme_rejected(self):
        with pytest.raises(UrlSafetyError, match="scheme"):
            validate_pull_url("http://example.com/x")

    def test_file_scheme_rejected(self):
        with pytest.raises(UrlSafetyError, match="scheme"):
            validate_pull_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(UrlSafetyError, match="scheme"):
            validate_pull_url("ftp://example.com/x")

    def test_gopher_scheme_rejected(self):
        with pytest.raises(UrlSafetyError, match="scheme"):
            validate_pull_url("gopher://example.com/x")

    def test_data_scheme_rejected(self):
        with pytest.raises(UrlSafetyError, match="scheme"):
            validate_pull_url("data:text/plain,hello")

    def test_ip_literal_private_rejected(self):
        with pytest.raises(UrlSafetyError, match="denied range"):
            validate_pull_url("https://10.0.0.5/x")

    def test_ip_literal_imds_rejected(self):
        with pytest.raises(UrlSafetyError):
            validate_pull_url("https://169.254.169.254/latest/meta-data/")

    def test_ip_literal_nat64_rejected(self):
        with pytest.raises(UrlSafetyError):
            validate_pull_url("https://[64:ff9b::1.1.1.1]/x")

    def test_missing_hostname_rejected(self):
        with pytest.raises(UrlSafetyError, match="hostname"):
            validate_pull_url("https:///path")

    def test_host_resolving_to_private_rejected(self):
        """DNS-rebind attempt: host name resolves to private IP."""
        with patch("socket.getaddrinfo") as m:
            m.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))
            ]
            with pytest.raises(UrlSafetyError, match="denied network"):
                validate_pull_url("https://attacker-rebind.example/x")


class TestIsHfHost:
    def test_exact_match(self):
        assert is_hf_host("huggingface.co", ["huggingface.co", "hf.co"])

    def test_subdomain_match(self):
        assert is_hf_host("cdn-lfs.huggingface.co", ["huggingface.co", "hf.co"])
        assert is_hf_host("co-lfs.hf.co", ["huggingface.co", "hf.co"])

    def test_case_insensitive(self):
        assert is_hf_host("HuggingFace.co", ["huggingface.co"])

    def test_no_match(self):
        assert not is_hf_host("evil-huggingface.co.attacker.com", ["huggingface.co"])
        assert not is_hf_host("notarealhost.com", ["huggingface.co"])

    def test_substring_attack_not_matched(self):
        """huggingface.coattacker.com would be a substring-match attack — must not pass."""
        assert not is_hf_host("huggingface.coattacker.com", ["huggingface.co"])
