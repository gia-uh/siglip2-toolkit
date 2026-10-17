"""Allow ``python -m siglip2_toolkit`` as an alias for the ``siglip2`` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
