# ABOUTME: Cleans MediaWiki 'wiki'-format extracts into plain narrative text for the corpus.
# ABOUTME: Removes the trailing block of citation/link sections while keeping same-named content sections.

import re

# Section titles that, at the end of an article, hold citations and links rather than narrative.
BOILERPLATE_SECTIONS = {
    "see also", "references", "notes", "citations", "sources", "bibliography",
    "further reading", "external links", "footnotes", "works cited",
    "primary sources", "general references", "explanatory notes",
}

HEADING = re.compile(r"^(=+)\s*(.+?)\s*=+\s*$", re.MULTILINE)

# Split a heading into the parts joined by "and", a comma, or an ampersand.
_TITLE_PARTS = re.compile(r"\s+and\s+|\s*[,&]\s*")


def _is_boilerplate(title):
    """True when a heading is built only from boilerplate section names.

    Handles combined appendix titles like "References and further reading"
    without keeping content sections like "Trade and commerce".
    """
    parts = [part.strip() for part in _TITLE_PARTS.split(title) if part.strip()]
    return bool(parts) and all(part in BOILERPLATE_SECTIONS for part in parts)


def clean_extract(text):
    """Turn a MediaWiki 'wiki'-format extract into clean narrative text.

    Drops the trailing block of citation/link sections, strips heading
    markers, and removes any leaked markup tags.
    """
    text = text[:_trailing_boilerplate_start(text)]
    text = HEADING.sub(r"\2", text)
    text = re.sub(r"<[a-zA-Z/][^>\n]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trailing_boilerplate_start(text):
    """Offset where the trailing run of boilerplate sections begins.

    Only the contiguous appendix at the end of the article is removed, so a
    content section that merely shares a name (e.g. an early 'Sources' section
    that discusses historiography) is preserved.
    """
    top_sections = [
        (match.start(), match.group(2).strip().lower())
        for match in HEADING.finditer(text)
        if len(match.group(1)) == 2
    ]
    cut = len(text)
    for start, title in reversed(top_sections):
        if not _is_boilerplate(title):
            break
        cut = start
    return cut
