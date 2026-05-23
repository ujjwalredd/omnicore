"""OmniCore-Local Infra MCP Server.

A decoupled, stdio-bound FastMCP server exposing system diagnostic tools to
the OmniCore gateway runtime. Communicates with the gateway over standard
input/output using the JSON-RPC 2.0 wire protocol — stdout is reserved
exclusively for protocol traffic; all human-readable diagnostics are routed
to stderr to avoid corrupting the channel.

Launch directly:
    python -m mcp_infra_server.server

Or attach from the gateway via a stdio MCP client transport.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time failure
    print(
        f"[OmniCore-MCP] FATAL: mcp[cli] package missing ({exc}). "
        "Install with: pip install 'mcp[cli]>=1.2.0'",
        file=sys.stderr,
        flush=True,
    )
    raise


SERVER_NAME: Final[str] = "OmniCore-Local-Infra"

# Hard deny-list of shell metacharacters / commands that must never be
# executed by a diagnostic tool, regardless of caller identity.
FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "rm", "sudo", "mv", "dd", "mkfs", "shutdown", "reboot", "halt",
    "kill", "killall", "passwd", "chown", "chmod", "curl", "wget",
    "nc", "ncat", "telnet", "ssh", "scp", "rsync", "eval", "exec",
    "source", ":(){",
)
FORBIDDEN_SYMBOLS: Final[tuple[str, ...]] = (
    ">", ">>", "<", "|", "&", "&&", "||", ";", "$(", "`", "\n", "\r",
)

# Safe allow-list of read-only diagnostic verbs accepted by the tool.
ALLOWED_COMMANDS: Final[frozenset[str]] = frozenset(
    {"uptime", "uname", "df", "free", "ps", "whoami", "hostname",
     "date", "id", "pwd", "echo", "ls", "cat"}
)

# Permitted target directories for read-only filesystem inspection.
ALLOWED_READ_PATHS: Final[tuple[str, ...]] = (
    "/proc", "/sys", "/tmp", os.path.expanduser("~"),
)


# -- Structured validation models ---------------------------------------------


class DiagnosticRequest(BaseModel):
    """Strict Pydantic model gating every diagnostic invocation."""

    server_id: str = Field(min_length=1, max_length=128)
    command: str = Field(min_length=1, max_length=512)

    @field_validator("server_id")
    @classmethod
    def _validate_server_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,128}", value):
            raise ValueError(
                "server_id must match [A-Za-z0-9_.-]{1,128}"
            )
        return value

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        lowered = value.lower()
        for sym in FORBIDDEN_SYMBOLS:
            if sym in value:
                raise ValueError(f"forbidden shell metacharacter: {sym!r}")
        try:
            tokens = shlex.split(value)
        except ValueError as exc:
            raise ValueError(f"unparseable command: {exc}") from exc
        if not tokens:
            raise ValueError("empty command after tokenization")
        verb = os.path.basename(tokens[0]).lower()
        for bad in FORBIDDEN_TOKENS:
            if bad in lowered.split() or bad == verb:
                raise ValueError(f"command verb {bad!r} is denied")
        if verb not in ALLOWED_COMMANDS:
            raise ValueError(
                f"command {verb!r} not in allow-list {sorted(ALLOWED_COMMANDS)}"
            )
        for tok in tokens[1:]:
            if tok.startswith("/") and not any(
                tok.startswith(p) for p in ALLOWED_READ_PATHS
            ):
                raise ValueError(
                    f"path {tok!r} outside permitted read roots"
                )
            if ".." in tok.split(os.sep):
                raise ValueError(f"directory traversal in token {tok!r}")
        return value


# -- FastMCP server bootstrap -------------------------------------------------


mcp = FastMCP(SERVER_NAME)


def _log_stderr(level: str, message: str) -> None:
    """Emit a structured log line to stderr (stdout is JSON-RPC reserved)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    sys.stderr.write(
        f"[{timestamp}] [{SERVER_NAME}] [{level}] {message}\n"
    )
    sys.stderr.flush()


@mcp.tool()
def execute_system_diagnostic(server_id: str, command: str) -> str:
    """Run a read-only diagnostic command against a logical local target.

    Args:
        server_id: Logical identifier of the target host/container/pod.
        command:   Allow-listed diagnostic verb plus arguments.

    Returns:
        Captured stdout of the diagnostic process, truncated to 8 KiB.

    Raises:
        ValueError: When Pydantic validation rejects the request.
        RuntimeError: When the underlying subprocess fails or times out.
    """
    try:
        request = DiagnosticRequest(server_id=server_id, command=command)
    except ValidationError as exc:
        _log_stderr("BLOCK", f"validation failed: {exc.errors()}")
        raise ValueError(f"diagnostic blocked by validation: {exc}") from exc

    tokens = shlex.split(request.command)
    _log_stderr(
        "EXEC",
        f"server_id={request.server_id} tokens={tokens}",
    )

    try:
        completed = subprocess.run(  # noqa: S603 - tokens are allow-list vetted
            tokens,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        _log_stderr("ERROR", f"binary missing for {tokens[0]!r}: {exc}")
        raise RuntimeError(f"diagnostic binary not found: {tokens[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        _log_stderr("ERROR", f"timeout running {tokens!r}")
        raise RuntimeError("diagnostic timed out after 10s") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        _log_stderr(
            "NONZERO",
            f"rc={completed.returncode} stderr={stderr[:256]!r}",
        )
    truncated = stdout[:8192]
    payload = (
        f"server_id={request.server_id}\n"
        f"exit_code={completed.returncode}\n"
        f"stdout:\n{truncated}\n"
        f"stderr:\n{stderr[:1024]}"
    )
    return payload


@mcp.tool()
def list_diagnostic_capabilities() -> str:
    """Return a JSON-ish description of permitted diagnostic verbs."""
    return (
        "{"
        f"\"server\": \"{SERVER_NAME}\", "
        f"\"allowed_commands\": {sorted(ALLOWED_COMMANDS)}, "
        f"\"allowed_read_paths\": {list(ALLOWED_READ_PATHS)}"
        "}"
    )


def main() -> None:
    """Start the FastMCP server bound to stdio transport."""
    _log_stderr("BOOT", f"starting {SERVER_NAME} on stdio transport")
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        _log_stderr("STOP", "interrupted by signal")
    except Exception as exc:  # pragma: no cover - top-level guard
        _log_stderr("FATAL", f"server crashed: {exc!r}")
        raise


if __name__ == "__main__":
    main()
