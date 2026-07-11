"""Turbohaul-Manager CLI entry.

Loads /etc/turbohaul/turbohaul.yaml (overridable via --config / TURBOHAUL_CONFIG_PATH),
applies TURBOHAUL_* env overrides per config.apply_env_overrides, and starts uvicorn.

For container deployment where binding 0.0.0.0 is needed, pass --allow-public-bind
or TURBOHAUL_ALLOW_PUBLIC_BIND=1. The yaml ServerConfig still validates as 127.0.0.1
; the public-bind override only changes the uvicorn host argument,
not the loaded BootConfig.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from turbohaul.api.main import create_app
from turbohaul.config import apply_env_overrides, load_config_yaml


log = logging.getLogger("turbohaul.main")


# uvicorn's access log floods docker-logs with ~1/sec health-poll
# `GET /status ... 200` lines (~22:1 vs the real decision logs), burying the
# per-turn decisions. Drop access records whose request path is EXACTLY a health
# endpoint. Safe-degrade: any unexpected record shape returns True (record kept),
# so this can never crash the log path or suppress a real request.
class _HealthPollAccessFilter(logging.Filter):
    _HEALTH_PATHS = frozenset({"/status", "/health", "/healthz"})

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn access record: args = (client_addr, method, full_path,
        # http_version, status_code) -> index 2 is the request target.
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        target = args[2]
        if not isinstance(target, str):
            return True
        path = target.split("?", 1)[0]  # drop ?query before exact-matching
        return path not in self._HEALTH_PATHS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turbohaul-manager",
        description="Ollama-shape inference manager.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("TURBOHAUL_CONFIG_PATH", "/etc/turbohaul/turbohaul.yaml")
        ),
        help="Path to turbohaul.yaml (default /etc/turbohaul/turbohaul.yaml).",
    )
    p.add_argument(
        "--allow-public-bind",
        action="store_true",
        default=os.environ.get("TURBOHAUL_ALLOW_PUBLIC_BIND") == "1",
        help=(
            "Override uvicorn host to 0.0.0.0 (container public bind). "
            "The default is 127.0.0.1; enable only inside an "
            "explicit network-policy boundary (e.g., a container with port mapping)."
        ),
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("TURBOHAUL_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper() if args.log_level != "trace" else "DEBUG",
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.config.exists():
        log.error("config not found: %s", args.config)
        return 2

    log.info("loading config: %s", args.config)
    cfg = apply_env_overrides(load_config_yaml(args.config))
    boot, runtime = cfg.split()

    bind_host = boot.server.host
    if args.allow_public_bind:
        bind_host = "::"  # noqa: S104 -- dual-stack container bind override (IPv6)
        log.warning(
            "--allow-public-bind in effect: uvicorn binding :: dual-stack "
            "(BootConfig.server.host=%s preserved)",
            boot.server.host,
        )

    log.info(
        "ready: %s:%d (ui.enabled=%s ui.static_path=%s)",
        bind_host,
        boot.server.port,
        boot.ui.enabled,
        boot.ui.static_path,
    )

    app = create_app(boot, runtime)
    # Silence /status /health /healthz poll spam on the access log.
    # Registered on the logger (not a handler); survives uvicorn's dictConfig
    # (disable_existing_loggers=False preserves logger-attached filters).
    logging.getLogger("uvicorn.access").addFilter(_HealthPollAccessFilter())
    _lvl = args.log_level if args.log_level != "trace" else "debug"
    if args.allow_public_bind:
        # httpx resolves the container IPv6 (AAAA) first and, unlike curl, does
        # NOT fall back to IPv4 -> an IPv4-only bind gives ConnectError. A bare "::"
        # uvicorn host goes IPv6-ONLY under docker (breaks the IPv4 host-port forward),
        # so bind an explicit dual-stack socket (IPV6_V6ONLY=0) serving BOTH families.
        import socket as _socket

        _sock = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        _sock.setsockopt(_socket.IPPROTO_IPV6, _socket.IPV6_V6ONLY, 0)
        _sock.bind(("::", boot.server.port))
        uvicorn.Server(
            uvicorn.Config(app, log_level=_lvl, access_log=True)
        ).run(sockets=[_sock])
    else:
        uvicorn.run(
            app,
            host=bind_host,
            port=boot.server.port,
            log_level=_lvl,
            access_log=True,
            # No --reload in production; per the deploy doctrine.
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
