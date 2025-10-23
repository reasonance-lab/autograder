import reflex as rx
from app.state import GradingState, GradingResult


def _report_card(result: GradingResult) -> rx.Component:
    """Displays a single student's grading report."""
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.el.h3(
                    result["student_file"].to_string(),
                    class_name="text-sm font-semibold text-gray-800 truncate",
                ),
                rx.el.div(
                    rx.el.span(result["grade"], class_name="text-sm font-bold"),
                    class_name=rx.cond(
                        result["grade"].contains("Error"),
                        "px-3 py-1 rounded-full bg-red-100 text-red-800",
                        "px-3 py-1 rounded-full bg-blue-100 text-blue-800",
                    ),
                ),
                class_name="flex justify-between items-center pb-3 border-b border-gray-200",
            ),
            rx.el.div(
                rx.markdown(
                    result["feedback"],
                    class_name="prose prose-sm max-w-none text-gray-600",
                ),
                class_name="mt-4 h-64 overflow-y-auto p-2 bg-gray-50 rounded-lg border",
            ),
            class_name="p-4",
        ),
        class_name="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-lg transition-all",
    )


def results_page() -> rx.Component:
    """Displays the grading results for all student papers."""
    return rx.el.main(
        rx.el.div(
            rx.el.div(
                rx.el.button(
                    rx.icon("arrow-left", class_name="mr-2"),
                    "Back to Upload",
                    on_click=rx.redirect("/"),
                    class_name="flex items-center text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors",
                ),
                class_name="w-full max-w-7xl mx-auto mb-6",
            ),
            rx.el.h1(
                "Grading Results",
                class_name="text-3xl font-bold tracking-tight text-gray-900",
            ),
            rx.el.p(
                f"Graded {GradingState.grading_results.length()} of {GradingState.student_paper_files.length()} papers.",
                class_name="mt-2 text-base text-gray-600",
            ),
            rx.cond(
                GradingState.grading_results.length() > 0,
                rx.el.div(
                    rx.foreach(GradingState.grading_results, _report_card),
                    class_name="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-8",
                ),
                rx.el.div(
                    rx.icon("search-x", class_name="w-12 h-12 text-gray-400"),
                    rx.el.h2(
                        "No Results Yet",
                        class_name="mt-4 text-lg font-semibold text-gray-700",
                    ),
                    rx.el.p(
                        "Grading has not been completed or no papers were processed.",
                        class_name="mt-1 text-sm text-gray-500",
                    ),
                    class_name="flex flex-col items-center justify-center text-center p-12 bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200 mt-8",
                ),
            ),
            class_name="w-full max-w-7xl mx-auto py-8 sm:py-12 px-4 sm:px-6 lg:px-8",
        ),
        class_name="font-['Inter'] bg-gray-50 min-h-screen",
    )