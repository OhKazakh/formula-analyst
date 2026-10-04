def listing(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count} {word if count == 1 else many or word + 's'}"


def lap_ranges(laps: list[int]) -> str:
    ranges, start = [], None
    for index, lap in enumerate(laps):
        start = lap if start is None else start
        if index + 1 == len(laps) or laps[index + 1] != lap + 1:
            ranges.append(str(lap) if start == lap else f"{start}–{lap}")
            start = None
    return ("lap " if len(laps) == 1 else "laps ") + listing(ranges)
