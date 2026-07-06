# ABOUTME: Tests for cleaning MediaWiki extracts into narrative text.
# ABOUTME: Guards trailing-section removal, early same-named content sections, marker and tag stripping.

from graphrag_wiki.wikitext import clean_extract


def test_strips_trailing_reference_sections():
    text = (
        "== Overview ==\nRome was big.\n\n"
        "== References ==\nSmith 2020.\n\n"
        "== External links ==\nhttp://example.org"
    )
    out = clean_extract(text)
    assert "Rome was big." in out
    assert "References" not in out
    assert "Smith 2020" not in out
    assert "External links" not in out


def test_keeps_early_content_section_sharing_a_boilerplate_name():
    # 'Sources' is a content section here, followed by real sections and a real appendix.
    text = (
        "== Sources ==\nThe main sources are Ammianus and others.\n\n"
        "== Organization ==\nThe army had many units.\n\n"
        "== References ==\nSmith 2020.\n"
    )
    out = clean_extract(text)
    assert "Ammianus" in out
    assert "The army had many units." in out
    assert "Smith 2020" not in out


def test_strips_trailing_combined_reference_heading():
    text = (
        "== Overview ==\nContent about the cult.\n\n"
        "== See also ==\nx\n\n"
        "== Notes ==\ny\n\n"
        "== References and further reading ==\nSmith 2020. ISBN 0-520-22067-6.\n"
    )
    out = clean_extract(text)
    assert "Content about the cult." in out
    assert "ISBN" not in out
    assert "References and further reading" not in out
    assert "See also" not in out


def test_keeps_content_section_joined_by_and():
    text = "== Trade and commerce ==\nGoods flowed widely.\n\n== References ==\nSmith 2020.\n"
    out = clean_extract(text)
    assert "Goods flowed widely." in out
    assert "Trade and commerce" in out
    assert "Smith 2020" not in out


def test_strips_heading_markers():
    out = clean_extract("== History ==\nText here.")
    assert "==" not in out
    assert "History" in out


def test_removes_leaked_tags():
    out = clean_extract("Communities <refnec>who struggled</refnec> with it.")
    assert "<refnec>" not in out
    assert "who struggled" in out


def test_article_without_appendix_keeps_all_content():
    out = clean_extract("== Intro ==\nAll content, no appendix here.")
    assert "All content, no appendix here." in out
