"""Check Python syntax without executing chatbot code or writing bytecode."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "chatbot.py", *sorted((root / "examples").rglob("*.py")), Path(__file__).resolve()]
    failures = []
    for path in paths:
        try:
            compile(path.read_bytes(), str(path.relative_to(root)), "exec")
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"{path.relative_to(root)}: {exc}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"PASS: {len(paths)} Python files compile; no chatbot code executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
