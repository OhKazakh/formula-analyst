def listing(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"
