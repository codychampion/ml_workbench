"""Logging and output formatting utilities."""


def print_header(title: str, width: int = 60):
    """Print formatted section header."""
    print(f"\n{'='*width}")
    print(title)
    print('='*width)


def print_section(title: str):
    """Print section marker (compact)."""
    print(f"\n{title}")
    print('-' * len(title))


def print_metrics(metrics: dict, indent: int = 0):
    """Print dict of metrics in aligned format."""
    indent_str = ' ' * indent
    max_key_len = max(len(str(k)) for k in metrics.keys())

    for key, value in metrics.items():
        key_str = str(key).ljust(max_key_len)
        if isinstance(value, float):
            print(f"{indent_str}{key_str}: {value:.4f}")
        else:
            print(f"{indent_str}{key_str}: {value}")


def print_step(step: str, status: str = "..."):
    """Print step with status."""
    print(f"  → {step} {status}")
