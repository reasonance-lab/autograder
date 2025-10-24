import reflex as rx
import random
import string
import os
import json
import logging
import anthropic
import re
import base64
from typing import TypedDict, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from markdown import markdown
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor
from html.parser import HTMLParser


def _format_chemical_formulas(text: str) -> str:
    """Converts chemical formulas like H2O to H<sub>2</sub>O for HTML rendering."""
    return re.sub("([A-Z][a-z]?)(\\d+)", "\\1<sub>\\2</sub>", text)


class GradingResult(TypedDict):
    student_file: str
    grade: str
    feedback: str
    report_file: str | None
    html_feedback: str


def _generate_unique_filename(name: str) -> str:
    """Generates a unique filename to avoid collisions."""
    random_prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{random_prefix}_{name}"


def _encode_pdf_to_base64(file_path: str) -> str:
    """Reads a PDF file and returns its base64-encoded representation."""
    try:
        with open(file_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except Exception as e:
        logging.exception(e)
        print(f"Error encoding {file_path} to base64: {e}")
        return ""


def _extract_json_from_markdown(text: str) -> str:
    """Extracts a JSON object from a string, handling multiple formats."""
    match = re.search("(?:json)?\\s*(\\{.*\\})\\s*", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1).strip()
        except json.JSONDecodeError as e:
            logging.exception(e)
            pass
    match = re.search("(\\{(?:[^{}]|\\{[^{}]*\\})*\\})", text)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1).strip()
        except json.JSONDecodeError as e:
            logging.exception(e)
            pass
    return ""


class HTMLToPDFParser(HTMLParser):
    def __init__(self, styles):
        super().__init__()
        self.styles = styles
        self.story = []
        self.tag_stack = []
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        self._process_text()
        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        self._process_text()
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag in ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]:
            self.story.append(Spacer(1, 0.1 * inch))

    def handle_data(self, data):
        self.current_text += data

    def handle_startendtag(self, tag, attrs):
        if tag == "hr":
            self._process_text()
            self.story.append(
                HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc"))
            )

    def _process_text(self):
        if not self.current_text.strip():
            self.current_text = ""
            return
        text = self.current_text
        style = self.styles["BodyText"]
        if "h1" in self.tag_stack:
            style = self.styles["h1"]
        elif "h2" in self.tag_stack:
            style = self.styles["h2"]
        elif "h3" in self.tag_stack:
            style = self.styles["h3"]
        elif "li" in self.tag_stack:
            text = f"• {text}"
            style.leftIndent = 20
        if "strong" in self.tag_stack or "b" in self.tag_stack:
            text = f"<b>{text}</b>"
        if "em" in self.tag_stack or "i" in self.tag_stack:
            text = f"<i>{text}</i>"
        text = re.sub("<sub>(.*?)</sub>", '<font size="7" rise="-2">\\1</font>', text)
        self.story.append(Paragraph(text, style))
        self.current_text = ""

    def get_story(self):
        self._process_text()
        return self.story


def _create_pdf_report(result: GradingResult) -> bytes:
    """Generates a PDF report from a grading result."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    html_feedback = markdown(result["feedback"])
    html_feedback_with_subs = _format_chemical_formulas(html_feedback)
    story = []
    story.append(Paragraph("Grading Report", styles["h1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(f"<b>Student File:</b> {result['student_file']}", styles["Normal"])
    )
    story.append(Paragraph(f"<b>Grade:</b> {result['grade']}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Feedback", styles["h2"]))
    story.append(Spacer(1, 0.1 * inch))
    parser = HTMLToPDFParser(styles)
    parser.feed(html_feedback_with_subs)
    story.extend(parser.get_story())
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


class GradingState(rx.State):
    """Manages the state for the automated exam grading application."""

    claude_models: list[str] = [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]
    selected_model: str = "claude-sonnet-4-5-20250929"
    grading_instructions: str = """Grade the AP chemistry test student responses per the rubric/answer key. 
Output: Table for the MCQ section of student answers compared to the answer key/rubric's answers and result (Correct/Wrong). For the FRQ section, output verbatim student answer followed by the answer key solution and evaluation of the student's answer solely based on the answer key solution. Supplement the evaluation with a grade. At the end of the report provide scores for MCQs, FRQs and total combined percentage."""
    answer_key_files: list[str] = []
    student_paper_files: list[str] = []
    is_grading: bool = False
    grading_progress: str = ""
    grading_results: list[GradingResult] = []
    grading_complete: bool = False
    html_feedback: str = ""

    @rx.event
    async def handle_answer_key_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of the answer key PDF."""
        if not files:
            return
        file = files[0]
        if file.content_type != "application/pdf":
            return rx.toast.error("Answer key must be a PDF file.")
        upload_data = await file.read()
        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        unique_filename = _generate_unique_filename(file.name)
        file_path = upload_dir / unique_filename
        with file_path.open("wb") as f:
            f.write(upload_data)
        self.answer_key_files = [unique_filename]
        return rx.toast.success("Answer key uploaded successfully.")

    @rx.event
    async def handle_student_papers_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of student paper PDFs."""
        if not files:
            return
        current_file_count = len(self.student_paper_files)
        if current_file_count + len(files) > 20:
            yield rx.toast.error(
                f"Cannot upload more than 20 student papers. You have {current_file_count} already."
            )
            return
        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_count = 0
        for file in files:
            if file.content_type != "application/pdf":
                yield rx.toast.warning(f"Skipping '{file.name}' as it is not a PDF.")
                continue
            upload_data = await file.read()
            unique_filename = _generate_unique_filename(file.name)
            file_path = upload_dir / unique_filename
            with file_path.open("wb") as f:
                f.write(upload_data)
            if unique_filename not in self.student_paper_files:
                self.student_paper_files.append(unique_filename)
                uploaded_count += 1
        if uploaded_count > 0:
            yield rx.toast.success(f"{uploaded_count} student paper(s) uploaded.")
            return

    @rx.event
    def clear_answer_key(self):
        """Clear the uploaded answer key."""
        self.answer_key_files = []
        self.grading_complete = False
        self.grading_results = []
        return rx.clear_selected_files("answer_key_upload")

    @rx.event
    def clear_student_papers(self):
        """Clear all uploaded student papers."""
        self.student_paper_files = []
        self.grading_complete = False
        self.grading_results = []
        return rx.clear_selected_files("student_papers_upload")

    @rx.event(background=True)
    async def start_grading(self):
        """Starts the grading process in the background."""
        async with self:
            self.is_grading = True
            self.grading_complete = False
            self.grading_results = []
            self.grading_progress = "Initializing..."
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                async with self:
                    self.is_grading = False
                yield rx.toast.error("ANTHROPIC_API_KEY not set.")
                return
            client = anthropic.Anthropic(api_key=api_key)
            upload_dir = rx.get_upload_dir()
            answer_key_path = upload_dir / self.answer_key_files[0]
            answer_key_base64 = _encode_pdf_to_base64(answer_key_path)
            if not answer_key_base64:
                async with self:
                    self.is_grading = False
                yield rx.toast.error("Could not process the answer key PDF.")
                return
            total_papers = len(self.student_paper_files)
            for i, student_file in enumerate(self.student_paper_files):
                async with self:
                    self.grading_progress = (
                        f"Processing student {i + 1} of {total_papers}..."
                    )
                yield
                student_paper_path = upload_dir / student_file
                student_paper_base64 = _encode_pdf_to_base64(student_paper_path)
                if not student_paper_base64:
                    result = GradingResult(
                        student_file=student_file,
                        grade="Error",
                        feedback="Could not process the student paper PDF.",
                        report_file=None,
                        html_feedback="<p>Could not process the student paper PDF.</p>",
                    )
                    async with self:
                        self.grading_results.append(result)
                    continue
                system_prompt = f"You are an expert teaching assistant. Grade the student's exam PDF based on the answer key PDF provided. The user will provide two PDF documents. \n                Provide a final grade and detailed, constructive feedback for each question. \n                {self.grading_instructions}\n\n                Format your response as a JSON object with two keys: 'grade' (a string like '85/100' or 'A-') and 'feedback' (a detailed markdown string).\n                "
                try:
                    message = (
                        client.messages.create(
                            model=self.selected_model,
                            max_tokens=4096,
                            system=system_prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "document",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": answer_key_base64,
                                            },
                                        },
                                        {
                                            "type": "document",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": student_paper_base64,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": "Please grade the student paper based on the answer key.",
                                        },
                                    ],
                                }
                            ],
                        )
                        .content[0]
                        .text
                    )
                    json_string = _extract_json_from_markdown(message)
                    if not json_string:
                        raise json.JSONDecodeError(
                            "No JSON found in response", message, 0
                        )
                    response_json = json.loads(json_string)
                    grade = response_json.get("grade", "Not found")
                    feedback = response_json.get("feedback", "No feedback provided.")
                    html_feedback = markdown(feedback)
                    html_feedback_with_subs = _format_chemical_formulas(html_feedback)
                    report = GradingResult(
                        student_file=student_file,
                        grade=grade,
                        feedback=feedback,
                        report_file=None,
                        html_feedback=html_feedback_with_subs,
                    )
                    async with self:
                        self.grading_results.append(report)
                except (json.JSONDecodeError, KeyError) as e:
                    logging.exception(e)
                    result = GradingResult(
                        student_file=student_file,
                        grade="API Error",
                        feedback=f"Failed to parse AI response: {e}",
                        report_file=None,
                        html_feedback=f"<p>Failed to parse AI response: {e}</p>",
                    )
                    async with self:
                        self.grading_results.append(result)
                except anthropic.AuthenticationError as e:
                    logging.exception(e)
                    async with self:
                        self.is_grading = False
                        self.grading_progress = "Authentication failed."
                    yield rx.toast.error(
                        "Invalid API key. Please check your ANTHROPIC_API_KEY.",
                        duration=10000,
                    )
                    return
                except anthropic.RateLimitError as e:
                    logging.exception(e)
                    async with self:
                        self.is_grading = False
                        self.grading_progress = "Rate limit exceeded."
                    yield rx.toast.error(
                        "API rate limit exceeded. Please wait and try again.",
                        duration=10000,
                    )
                    return
                except anthropic.APIConnectionError as e:
                    logging.exception(e)
                    async with self:
                        self.is_grading = False
                        self.grading_progress = "Network error."
                    yield rx.toast.error(
                        "Network error: Could not connect to Anthropic API.",
                        duration=10000,
                    )
                    return
                except Exception as e:
                    logging.exception(e)
                    result = GradingResult(
                        student_file=student_file,
                        grade="API Error",
                        feedback=f"An error occurred with the Anthropic API: {e}",
                        report_file=None,
                        html_feedback=f"<p>An error occurred with the Anthropic API: {e}</p>",
                    )
                    async with self:
                        self.grading_results.append(result)
            async with self:
                self.grading_complete = True
                self.is_grading = False
                self.grading_progress = "Grading complete!"
            yield rx.toast.success("All papers graded successfully!")
        except Exception as e:
            logging.exception(e)
            async with self:
                self.is_grading = False
                self.grading_progress = "An unexpected error occurred."
            yield rx.toast.error(f"Grading failed: {e}")

    @rx.event
    def download_report(self, result: GradingResult):
        """Generate and download a PDF report for a single result."""
        try:
            pdf_data = _create_pdf_report(result)
            report_filename = f"report_{result['student_file'].replace('.pdf', '')}.pdf"
            return rx.download(data=pdf_data, filename=report_filename)
        except Exception as e:
            logging.exception(e)
            return rx.toast.error(f"Failed to generate PDF report: {e}")

    @rx.event
    def download_html_report(self, result: GradingResult):
        """Generate and download an HTML report for a single result."""
        try:
            html_content = f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n    <meta charset='UTF-8'>\n    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n    <title>Grading Report: {result['student_file']}</title>\n    <style>\n        body {{ font-family: sans-serif; line-height: 1.6; padding: 2em; max-width: 800px; margin: auto; color: #333; }}\n        h1, h2, h3 {{ color: #222; }}\n        pre {{ background-color: #f4f4f4; padding: 1em; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }}\n        code {{ font-family: monospace; }}\n        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}\n        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}\n        th {{ background-color: #f2f2f2; }}\n        sub {{ font-size: 0.75em; line-height: 0; position: relative; vertical-align: baseline; bottom: -0.25em; }}\n    </style>\n</head>\n<body>\n    <h1>Grading Report</h1>\n    <p><b>Student File:</b> {result['student_file']}</p>\n    <p><b>Grade:</b> {result['grade']}</p>\n    <hr>\n    <h2>Feedback</h2>\n    {result['html_feedback']}\n</body>\n</html>"
            report_filename = (
                f"report_{result['student_file'].replace('.pdf', '')}.html"
            )
            return rx.download(data=html_content, filename=report_filename)
        except Exception as e:
            logging.exception(e)
            return rx.toast.error(f"Failed to generate HTML report: {e}")

    @rx.var
    def can_start_grading(self) -> bool:
        """Check if grading can be started."""
        return (
            len(self.answer_key_files) == 1
            and len(self.student_paper_files) > 0
            and (not self.is_grading)
        )