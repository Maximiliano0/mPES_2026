"""
Terminal formatting utilities for improved console output aesthetics.
Provides functions for formatted messages, progress indicators, and styled text.
"""

def header(text, width=80):
    """Print a formatted header with border."""
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")

def section(text, width=80):
    """Print a formatted section title."""
    print(f"\n{'-'*width}")
    print(f"  {text}")
    print(f"{'-'*width}\n")

def success(text, prefix="✓"):
    """Print a success message."""
    print(f"{prefix} {text}")

def error(text, prefix="❌"):
    """Print an error message."""
    print(f"{prefix} {text}")

def info(text, prefix="ℹ"):
    """Print an info message."""
    print(f"{prefix} {text}")

def progress(current, total, text="Progress", width=40):
    """Print a progress bar."""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    print(f"\r{text}: [{bar}] {pct}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()

def list_item(text, level=1, prefix="→"):
    """Print a list item with indentation."""
    indent = "  " * (level - 1)
    print(f"{indent}{prefix} {text}")

def data_row(label, value, label_width=40, value_width=35):
    """Print a formatted data row."""
    print(f"{label:<{label_width}} {value:<{value_width}}")

def separator(width=80, char="="):
    """Print a separator line."""
    print(f"{char*width}")

def banner(text, width=80, char="="):
    """Print a centered banner."""
    padding = (width - len(text) - 2) // 2
    print(f"{char*width}")
    print(f"{char} {text.center(width-4)} {char}")
    print(f"{char*width}\n")

def box(text, width=80):
    """Print text in a box."""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines) if lines else 0
    box_width = min(max(max_len + 4, width), width)
    
    print(f"┌{'─'*(box_width-2)}┐")
    for line in lines:
        padding = box_width - 4 - len(line)
        print(f"│ {line}{' '*padding} │")
    print(f"└{'─'*(box_width-2)}┘\n")

def highlight(text, width=80):
    """Print highlighted text."""
    print(f"▶ {text}")
