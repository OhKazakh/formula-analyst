def listing(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count} {word if count == 1 else many or word + 's'}"
