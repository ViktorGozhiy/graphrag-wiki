# ABOUTME: Recursive text chunker producing the shared retrieval unit for vector, keyword, and graph search.
# ABOUTME: Splits on paragraph then sentence then word boundaries, packs to a size budget, adds overlap.

SEPARATORS = ("\n\n", ". ", " ", "")


def chunk_text(text, chunk_size=1600, overlap=240):
    """Split text into overlapping chunks that respect natural boundaries.

    Each chunk stays within chunk_size (plus the overlap carried from the
    previous chunk). Splitting prefers paragraph, then sentence, then word
    boundaries, falling back to a hard character cut only when a single token
    exceeds chunk_size.
    """
    units = _atomic_units(text, chunk_size)
    groups = _pack(units, chunk_size)

    chunks = []
    for index, group in enumerate(groups):
        body = "\n\n".join(group)
        if index > 0:
            prefix = _overlap_tail("\n\n".join(groups[index - 1]), overlap)
            if prefix:
                body = f"{prefix}\n\n{body}"
        chunks.append(body)
    return chunks


def _atomic_units(text, chunk_size):
    """Break text into pieces that each fit within chunk_size."""
    units = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_long(paragraph, chunk_size, SEPARATORS[1:]))
    return units


def _split_long(text, chunk_size, separators):
    """Greedily pack sub-pieces up to chunk_size, recursing on finer separators."""
    if len(text) <= chunk_size:
        return [text]
    separator = separators[0]
    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    units = []
    buffer = ""
    for piece in text.split(separator):
        candidate = f"{buffer}{separator}{piece}" if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            units.append(buffer)
        if len(piece) <= chunk_size:
            buffer = piece
        else:
            units.extend(_split_long(piece, chunk_size, separators[1:]))
            buffer = ""
    if buffer:
        units.append(buffer)
    return units


def _pack(units, chunk_size):
    """Group consecutive units so each group's joined length fits chunk_size."""
    groups = []
    current = []
    current_len = 0
    for unit in units:
        addition = len(unit) + (2 if current else 0)
        if current and current_len + addition > chunk_size:
            groups.append(current)
            current = []
            current_len = 0
            addition = len(unit)
        current.append(unit)
        current_len += addition
    if current:
        groups.append(current)
    return groups


def _overlap_tail(text, overlap):
    """Return up to `overlap` trailing characters, starting at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail
