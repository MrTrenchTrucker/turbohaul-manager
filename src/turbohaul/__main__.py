"""Turbohaul-Manager CLI entry.

Loads /etc/turbohaul/turbohaul.yaml (overridable via --config / TURBOHAUL_CONFIG_PATH),
applies TURBOHAUL_* env overrides per config.apply_env_overrides, and starts uvicorn.

For container deployment where binding 0.0.0.0 is needed, pass --allow-public-bind
or TURBOHAUL_ALLOW_PUBLIC_BIND=1. The yaml ServerConfig still validates as 127.0.0.1
(per v0.2 §3.2); the public-bind override only changes the uvicorn host argument,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turbohaul-manager",
        description="Ollama-shape inference manager (v0.2).",
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
            "v0.2 §3.2 internal-network default is 127.0.0.1; enable only inside an "
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
        bind_host = "0.0.0.0"  # noqa: S104 -- explicit container bind override
        log.warning(
            "--allow-public-bind in effect: uvicorn binding 0.0.0.0 "
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
    uvicorn.run(
        app,
        host=bind_host,
        port=boot.server.port,
        log_level=args.log_level if args.log_level != "trace" else "debug",
        access_log=True,
        # No --reload in production; v0.2 §13 + post- deploy doctrine.
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
