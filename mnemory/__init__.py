"""mnemory — A self-hosted, two-tier memory system for AI agents and assistants."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mnemory")
except PackageNotFoundError:
    # Package not installed (running from source without pip install)
    __version__ = "0.0.0+dev"
