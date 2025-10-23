import reflex as rx
import random
import string
import os
import json
import logging
import anthropic
import re
from pypdf import PdfReader
from typing import TypedDict, Any


class GradingResult(TypedDict):
    student_file: str
    grade: str
    feedback: str
    report_file: str | None


def _generate_unique_filename(name: str) -> str:
    """Generates a unique filename to avoid collisions."""
    random_prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{random_prefix}_{name}"


def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = "".join((page.extract_text() for page in reader.pages))
        return text
    except Exception as e:
        logging.exception(e)
        print(f"Error extracting text from {file_path}: {e}")
        return ""


def _extract_json_from_markdown(markdown_string: str) -> str:
    """Extracts a JSON object from a markdown code block."""
    match = re.search("(?:json)?\\s*(\\{.*?\\})\\s*", markdown_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    if markdown_string.strip().startswith("{"):
        return markdown_string.strip()
    return ""


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
    grading_instructions: str = ""
    answer_key_files: list[str] = []
    student_paper_files: list[str] = []
    is_grading: bool = False
    grading_progress: str = ""
    grading_results: list[GradingResult] = []
    grading_complete: bool = False

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
            answer_key_text = _extract_text_from_pdf(answer_key_path)
            if not answer_key_text:
                async with self:
                    self.is_grading = False
                yield rx.toast.error("Could not extract text from answer key.")
                return
            total_papers = len(self.student_paper_files)
            for i, student_file in enumerate(self.student_paper_files):
                async with self:
                    self.grading_progress = (
                        f"Processing student {i + 1} of {total_papers}..."
                    )
                yield
                student_paper_path = upload_dir / student_file
                student_paper_text = _extract_text_from_pdf(student_paper_path)
                if not student_paper_text:
                    result = GradingResult(
                        student_file=student_file,
                        grade="Error",
                        feedback="Could not extract text from PDF.",
                        report_file=None,
                    )
                    async with self:
                        self.grading_results.append(result)
                    continue
                system_prompt = f"You are an expert teaching assistant. Grade the following student exam based on the provided answer key. \n                Provide a final grade and detailed, constructive feedback for each question. \n                {self.grading_instructions}\n\n                Format your response as a JSON object with two keys: 'grade' (a string like '85/100' or 'A-') and 'feedback' (a detailed markdown string).\n                "
                try:
                    message = (
                        client.messages.create(
                            model=self.selected_model,
                            max_tokens=2048,
                            system=system_prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Answer Key:\n\n{answer_key_text}\n\n---\n\nStudent Paper:\n\n{student_paper_text}",
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
                    report = GradingResult(
                        student_file=student_file,
                        grade=grade,
                        feedback=feedback,
                        report_file=None,
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
                    )
                    async with self:
                        self.grading_results.append(result)
                except Exception as e:
                    logging.exception(e)
                    result = GradingResult(
                        student_file=student_file,
                        grade="API Error",
                        feedback=f"An error occurred with the Anthropic API: {e}",
                        report_file=None,
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

    @rx.var
    def can_start_grading(self) -> bool:
        """Check if grading can be started."""
        return (
            len(self.answer_key_files) == 1
            and len(self.student_paper_files) > 0
            and (not self.is_grading)
        )