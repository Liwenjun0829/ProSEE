"""Command-line entry point for the preserved symbolic-regression pipeline."""

from ._core import engine


def main() -> None:
    engine.main()


if __name__ == "__main__":
    main()
