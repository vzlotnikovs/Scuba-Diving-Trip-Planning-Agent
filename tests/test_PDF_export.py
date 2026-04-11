"""Tests for ``Agent.PDF_export`` helpers and PDF generation."""

from reportlab.lib.styles import getSampleStyleSheet

from Agent.PDF_export import (
    _content_to_flowables,
    _escape_reportlab,
    _markdown_bold_to_reportlab,
    create_pdf_chat,
)


def test_escape_reportlab_escapes_xml_specials() -> None:
    assert _escape_reportlab("a & b < c > d") == "a &amp; b &lt; c &gt; d"


def test_markdown_bold_to_reportlab() -> None:
    assert _markdown_bold_to_reportlab("Hello **world**") == "Hello <b>world</b>"


def test_content_to_flowables_empty_yields_placeholder() -> None:
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    flowables = _content_to_flowables("   \n  ", style)
    assert len(flowables) >= 1


def test_content_to_flowables_bullet_and_bold() -> None:
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    flowables = _content_to_flowables("- Item with **bold**", style)
    assert len(flowables) >= 1


def test_create_pdf_chat_returns_pdf_bytes() -> None:
    messages = [
        {"role": "user", "content": "Hello **diver**"},
        {"role": "assistant", "content": "Welcome to the plan."},
    ]
    pdf_bytes = create_pdf_chat(messages)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
