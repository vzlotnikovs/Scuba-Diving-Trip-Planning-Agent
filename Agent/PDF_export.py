import os
import re
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Sequence, cast
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Flowable
from reportlab.lib.colors import blue, green

from constants import PAGE_TITLE_2


def _escape_reportlab(text: str) -> str:
    """Escape special characters for safe inclusion in ReportLab XML.

    Converts ampersands and angle brackets to their corresponding HTML entities.

    Args:
        text (str): The raw string to escape.

    Returns:
        str: The escaped string safe for ReportLab formatting.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_bold_to_reportlab(text: str) -> str:
    """Convert Markdown bold syntax to ReportLab HTML-like bold tags.

    Args:
        text (str): The string containing markdown bold syntax (e.g., **text**).

    Returns:
        str: The string with bold markdown replaced by <b>text</b> tags.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _content_to_flowables(content: str, style: ParagraphStyle) -> Sequence[Flowable]:
    """Convert message text into a sequence of ReportLab Flowables.

    Processes line breaks, paragraphs, bullet points, and bold text formatting,
    returning ReportLab elements ready for the PDF builder.

    Args:
        content (str): The chat message content string.
        style (ParagraphStyle): The base paragraph style to apply to text.

    Returns:
        Sequence[Flowable]: A list of ReportLab Flowable objects representing
            the formatted message.
    """
    if not content.strip():
        return [Paragraph(" ", style)]
    escaped = _escape_reportlab(content)
    flowables: list[Flowable] = []
    for line in escaped.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            flowables.append(Paragraph("<br/>", style))
            continue
        if line_stripped.startswith("- ") or line_stripped.startswith("* "):
            bullet_text = line_stripped[2:].strip()
            bullet_text = _markdown_bold_to_reportlab(bullet_text)
            flowables.append(
                Paragraph(
                    f'<bullet bulletAnchor="start">&bull;</bullet> {bullet_text}',
                    style,
                )
            )
        else:
            line_with_bold = _markdown_bold_to_reportlab(line_stripped)
            flowables.append(Paragraph(line_with_bold, style))
    return flowables


def _register_emoji_font() -> str:
    """Register an emoji-capable TrueType font for ReportLab.

    Attempts to find and register fonts like Symbola or Segoe UI Emoji to
    support rendering emojis in PDF chat exports.

    Returns:
        str: The name of the registered font if successful, or "Helvetica"
            as a fallback.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        _dir = os.path.dirname(__file__)
        candidates = [
            ("Symbola", os.path.join(_dir, "fonts", "Symbola.ttf")),
            ("Symbola", os.path.join(_dir, "..", "fonts", "Symbola.ttf")),
            ("Segoe UI Emoji", os.path.expandvars(r"%windir%\Fonts\seguiemj.ttf")),
        ]
        for font_name, path in candidates:
            if os.path.isfile(path):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, path))
                    return font_name
                except Exception:
                    continue
        return "Helvetica"
    except Exception:
        return "Helvetica"


@st.cache_data
def create_pdf_chat(messages: List[Dict[str, str]]) -> bytes:
    """Export chat history to a formatted PDF document.

    Builds a PDF containing all chat messages, distinguishing between the user
    and the assistant with distinct headers and fonts. Handles emoji support
    by dynamically applying custom fonts if available.

    Args:
        messages (List[Dict[str, str]]): The conversation history list.

    Returns:
        bytes: The binary content of the generated PDF document.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    body_font = _register_emoji_font()
    normal_style = cast(ParagraphStyle, styles["Normal"])

    if body_font != "Helvetica":
        styles.add(
            ParagraphStyle(
                name="NormalEmoji",
                parent=normal_style,
                fontName=body_font,
                fontSize=normal_style.fontSize,
            )
        )
        content_style = cast(ParagraphStyle, styles["NormalEmoji"])
    else:
        content_style = normal_style

    story: list[Flowable] = []
    header_font = "Helvetica-Bold" if body_font == "Helvetica" else body_font
    styles.add(
        ParagraphStyle(
            name="UserHeader",
            fontSize=12,
            fontName=header_font,
            textColor=blue,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AgentHeader",
            fontSize=12,
            fontName=header_font,
            textColor=green,
        )
    )

    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            story.append(
                Paragraph(
                    f"<b>👤 You ({datetime.now().strftime('%H:%M')}):</b>",
                    styles["UserHeader"],
                )
            )
        else:
            story.append(
                Paragraph(
                    f"<b>{PAGE_TITLE_2} ({datetime.now().strftime('%H:%M')}):</b>",
                    styles["AgentHeader"],
                )
            )
        story.extend(_content_to_flowables(content, content_style))
        story.append(Spacer(1, 12))

    try:
        doc.build(story)
    except Exception as e:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        doc.build([Paragraph(f"Export error: {e}", styles["Normal"])])

    buffer.seek(0)
    return buffer.getvalue()
