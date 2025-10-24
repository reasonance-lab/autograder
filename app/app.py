import reflex as rx
from app.state import GradingState, GradingResult
from app.results import results_page


def _upload_area(
    title: str,
    description: str,
    upload_id: str,
    on_upload: rx.event.EventHandler,
    on_clear: rx.event.EventHandler,
    uploaded_files: rx.Var[list[str]],
    accept: dict,
    multiple: bool,
    max_files: int,
) -> rx.Component:
    """Creates a styled file upload area."""
    return rx.el.div(
        rx.el.div(
            rx.icon("file-text", class_name="w-6 h-6 text-blue-500"),
            class_name="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-lg border border-blue-200",
        ),
        rx.el.h3(title, class_name="mt-4 text-sm font-semibold text-gray-900"),
        rx.el.p(description, class_name="mt-1 text-xs text-gray-600"),
        rx.upload.root(
            rx.el.div(
                rx.icon("cloud_upload", class_name="w-8 h-8 text-gray-400"),
                rx.el.p(
                    "Drag & drop or click to upload",
                    class_name="text-sm text-gray-500 mt-2",
                ),
                class_name="flex flex-col items-center justify-center w-full h-32 mt-4 border-2 border-dashed border-gray-300 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors",
            ),
            id=upload_id,
            accept=accept,
            multiple=multiple,
            max_files=max_files,
            on_drop=on_upload,
            class_name="w-full cursor-pointer mt-4",
        ),
        rx.cond(
            uploaded_files.length() > 0,
            rx.el.div(
                rx.el.div(
                    rx.foreach(
                        uploaded_files,
                        lambda file: rx.el.div(
                            rx.icon("file", class_name="w-4 h-4 text-gray-500"),
                            rx.el.span(file, class_name="truncate"),
                            class_name="flex items-center gap-2 bg-white px-3 py-2 rounded-md border border-gray-200 text-xs text-gray-700",
                        ),
                    ),
                    class_name="flex flex-wrap gap-2 mt-4",
                ),
                rx.el.button(
                    "Clear",
                    on_click=on_clear,
                    class_name="mt-2 text-xs text-blue-600 hover:underline",
                ),
                class_name="w-full",
            ),
            None,
        ),
        class_name="flex flex-col items-center p-6 bg-white rounded-2xl border border-gray-200 shadow-sm transition-all hover:shadow-md",
    )


def _grading_overlay() -> rx.Component:
    return rx.cond(
        GradingState.is_grading,
        rx.el.div(
            rx.el.div(
                rx.spinner(class_name="w-8 h-8 text-white"),
                rx.el.h2(
                    "Grading in Progress",
                    class_name="text-xl font-semibold text-white mt-4",
                ),
                rx.el.p(GradingState.grading_progress, class_name="text-gray-300 mt-2"),
                class_name="flex flex-col items-center justify-center bg-gray-800/80 p-8 rounded-2xl shadow-lg",
            ),
            class_name="absolute inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
        ),
    )


def index() -> rx.Component:
    """The main page for the Automated Exam Grading application."""
    return rx.el.main(
        _grading_overlay(),
        rx.el.div(
            rx.el.div(
                rx.icon("bot", class_name="w-8 h-8 text-white"),
                class_name="flex items-center justify-center w-16 h-16 bg-blue-600 rounded-2xl shadow-lg",
            ),
            rx.el.h1(
                "Automated Exam Grading",
                class_name="mt-6 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl",
            ),
            rx.el.p(
                "Upload your answer key and student papers to start grading with AI.",
                class_name="mt-4 text-base text-gray-600",
            ),
            rx.el.div(
                rx.el.label(
                    "Select Claude Model",
                    class_name="block text-sm font-medium text-gray-700 mb-2",
                ),
                rx.el.select(
                    rx.foreach(
                        GradingState.claude_models,
                        lambda model: rx.el.option(model, value=model),
                    ),
                    value=GradingState.selected_model,
                    on_change=GradingState.set_selected_model,
                    class_name="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all",
                ),
                class_name="w-full mt-8",
            ),
            rx.el.div(
                _upload_area(
                    title="Answer Key PDF",
                    description="Upload the single master answer key.",
                    upload_id="answer_key_upload",
                    on_upload=GradingState.handle_answer_key_upload,
                    on_clear=GradingState.clear_answer_key,
                    uploaded_files=GradingState.answer_key_files,
                    accept={"application/pdf": [".pdf"]},
                    multiple=False,
                    max_files=1,
                ),
                _upload_area(
                    title="Student Papers",
                    description=f"Upload up to 20 student papers. ({GradingState.student_paper_files.length()}/20)",
                    upload_id="student_papers_upload",
                    on_upload=GradingState.handle_student_papers_upload,
                    on_clear=GradingState.clear_student_papers,
                    uploaded_files=GradingState.student_paper_files,
                    accept={"application/pdf": [".pdf"]},
                    multiple=True,
                    max_files=20,
                ),
                class_name="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8 w-full",
            ),
            rx.el.div(
                rx.el.label(
                    "Optional Grading Instructions",
                    class_name="block text-sm font-medium text-gray-700 mb-2",
                ),
                rx.el.textarea(
                    default_value=GradingState.grading_instructions,
                    on_change=GradingState.set_grading_instructions,
                    placeholder="e.g., 'Be lenient on spelling mistakes.', 'Award partial credit for showing work.'...",
                    class_name="w-full h-24 px-4 py-3 bg-white border border-gray-300 rounded-xl shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all",
                ),
                class_name="w-full mt-8",
            ),
            rx.cond(
                GradingState.grading_complete,
                rx.el.button(
                    rx.icon("check_check", class_name="mr-2"),
                    "View Results",
                    on_click=rx.redirect("/results"),
                    class_name="w-full mt-8 px-6 py-4 flex items-center justify-center text-base font-semibold text-white bg-green-600 rounded-xl shadow-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all",
                ),
                rx.el.button(
                    rx.icon("play", class_name="mr-2"),
                    "Start Grading",
                    on_click=GradingState.start_grading,
                    disabled=~GradingState.can_start_grading,
                    class_name="w-full mt-8 px-6 py-4 flex items-center justify-center text-base font-semibold text-white bg-blue-600 rounded-xl shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed",
                ),
            ),
            class_name="w-full max-w-4xl flex flex-col items-center text-center p-6 sm:p-8",
        ),
        class_name="font-['Inter'] bg-gray-50 min-h-screen flex items-center justify-center",
    )


app = rx.App(
    theme=rx.theme(appearance="light"),
    head_components=[
        rx.el.link(rel="preconnect", href="https://fonts.googleapis.com"),
        rx.el.link(rel="preconnect", href="https://fonts.gstatic.com", cross_origin=""),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
            rel="stylesheet",
        ),
        rx.el.style("""
            /* Enhanced subscript rendering for chemical formulas */
            sub {
                font-size: 0.7em;
                line-height: 0;
                position: relative;
                vertical-align: baseline;
                bottom: -0.25em;
                font-variant-numeric: normal;
            }

            /* Ensure subscripts render properly in prose content */
            .prose sub {
                font-size: 0.7em;
                line-height: 0;
                position: relative;
                vertical-align: baseline;
                bottom: -0.25em;
            }

            /* Better rendering for chemical formulas */
            .prose {
                font-feature-settings: 'subs' 1;
            }
        """),
    ],
)
app.add_page(index, title="Automated Exam Grading")
app.add_page(results_page, route="/results", title="Grading Results")