"""
State management for LLM Cross-Talk Analyzer with improved convergence logic.
"""
import reflex as rx
import os
import asyncio
import openai
import anthropic
from typing import Literal, TypedDict, cast
import logging
from collections import Counter
import math

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelResponse(TypedDict):
    """Type definition for a single iteration response."""
    iteration: int
    openai_response: str
    claude_response: str
    similarity: float
    change_rate: float


class ComparisonState(rx.State):
    """Manages the state for the LLM comparison application with enhanced convergence."""

    # Core state
    prompt: str = ""
    is_loading: bool = False
    is_iterating: bool = False
    automated_running: bool = False
    converged: bool = False
    mode: Literal["manual", "automated"] = "manual"
    history: list[ModelResponse] = []
    
    # Configuration state
    convergence_threshold: float = 0.90
    max_iterations: int = 10
    openai_model: str = "gpt-3.5-turbo"
    claude_model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    
    # Advanced settings visibility
    show_settings: bool = False
    
    # Private clients
    _openai_client: openai.OpenAI | None = None
    _anthropic_client: anthropic.Anthropic | None = None

    def _initialize_clients(self):
        """Initializes API clients if they don't exist."""
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

    @rx.var
    def has_responses(self) -> bool:
        """Check if there is any history to display."""
        return len(self.history) > 0

    @rx.var
    def last_similarity_score(self) -> float:
        """Returns the similarity score of the last iteration as percentage."""
        if not self.history:
            return 0.0
        return self.history[-1]["similarity"] * 100

    @rx.var
    def current_iteration_count(self) -> int:
        """Returns the current iteration count."""
        return len(self.history)

    @rx.var
    def convergence_status(self) -> str:
        """Returns a human-readable convergence status."""
        if not self.history:
            return "No iterations yet"
        
        last_similarity = self.history[-1]["similarity"]
        threshold_pct = self.convergence_threshold * 100
        
        if last_similarity >= self.convergence_threshold:
            return f"✓ Converged at {last_similarity * 100:.1f}% similarity"
        elif len(self.history) >= self.max_iterations:
            return f"Max iterations reached ({self.max_iterations})"
        else:
            gap = (self.convergence_threshold - last_similarity) * 100
            return f"Need {gap:.1f}% more to reach {threshold_pct:.0f}% threshold"

    @rx.event
    def set_mode(self, new_mode: Literal["manual", "automated"]):
        """Set the evaluation mode."""
        self.mode = new_mode

    @rx.event
    def toggle_settings(self):
        """Toggle settings panel visibility."""
        self.show_settings = not self.show_settings

    @rx.event
    def update_convergence_threshold(self, value: list[float]):
        """Update convergence threshold from slider."""
        if value and len(value) > 0:
            self.convergence_threshold = value[0]

    @rx.event
    def update_max_iterations(self, value: list[int]):
        """Update max iterations from slider."""
        if value and len(value) > 0:
            self.max_iterations = int(value[0])

    @rx.event
    def update_temperature(self, value: list[float]):
        """Update temperature from slider."""
        if value and len(value) > 0:
            self.temperature = value[0]

    @rx.event
    def clear_all(self):
        """Clear all history and reset state."""
        self.history = []
        self.converged = False
        self.automated_running = False
        self.is_iterating = False
        self.prompt = ""

    @rx.event(background=True)
    async def get_initial_responses(self, form_data: dict):
        """Fetches initial responses from both OpenAI and Claude."""
        async with self:
            self.prompt = form_data.get("prompt", "")
            if not self.prompt or not self.prompt.strip():
                yield rx.toast.error("Please enter a prompt.", duration=3000)
                return
            
            self.is_loading = True
            self.is_iterating = False
            self.automated_running = False
            self.converged = False
            self.history = []
            self._initialize_clients()

        try:
            openai_task = self._fetch_openai(self.prompt)
            claude_task = self._fetch_claude(self.prompt)
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)

            if openai_res is None or claude_res is None:
                async with self:
                    self.is_loading = False
                if openai_res is None:
                    yield rx.toast.error("OpenAI failed to respond.", duration=5000)
                if claude_res is None:
                    yield rx.toast.error("Claude failed to respond.", duration=5000)
                return

            similarity = self._cosine_similarity(openai_res, claude_res)

            async with self:
                initial_responses: ModelResponse = {
                    "iteration": 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                    "change_rate": 0.0,  # First iteration has no change
                }
                self.history.append(initial_responses)
                
                # Auto-start if in automated mode
                if self.mode == "automated":
                    yield ComparisonState.run_automated_cycle

        except Exception as e:
            logging.exception(f"Error fetching initial responses: {e}")
            yield rx.toast.error(f"An unexpected error occurred: {e}", duration=5000)
        finally:
            async with self:
                self.is_loading = False

    @rx.event(background=True)
    async def iterate_manual_mode(self):
        """Performs one round of cross-evaluation in manual mode."""
        async with self:
            if not self.history:
                return
            
            self.is_iterating = True
            self._initialize_clients()
            
            last_iteration = self.history[-1]
            openai_previous = last_iteration["openai_response"]
            claude_previous = last_iteration["claude_response"]

            # Generic prompt template (domain-agnostic)
            new_prompt_template = """Original User Request:
{original_prompt}

Your Previous Response:
---
{your_previous_response}
---

Alternative Response from Another Model:
---
{other_model_response}
---

Instructions:
- Carefully review BOTH responses against the original user request
- Identify the strengths and weaknesses of each approach
- If the alternative response has superior reasoning, insights, or completeness, integrate those elements
- Eliminate purely stylistic differences (formatting, phrasing variations that don't affect meaning)
- Synthesize the best aspects of both responses into a refined answer
- DO NOT add meta-commentary about the comparison process
- Output ONLY your refined, complete response to the original user request"""

            openai_new_prompt = new_prompt_template.format(
                original_prompt=self.prompt,
                your_previous_response=openai_previous,
                other_model_response=claude_previous,
            )
            claude_new_prompt = new_prompt_template.format(
                original_prompt=self.prompt,
                your_previous_response=claude_previous,
                other_model_response=openai_previous,
            )

        try:
            openai_task = self._fetch_openai(openai_new_prompt)
            claude_task = self._fetch_claude(claude_new_prompt)
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)

            async with self:
                if openai_res is None or claude_res is None:
                    self.is_iterating = False
                    if openai_res is None:
                        yield rx.toast.error(
                            "OpenAI failed to respond during iteration.", duration=5000
                        )
                    if claude_res is None:
                        yield rx.toast.error(
                            "Claude failed to respond during iteration.", duration=5000
                        )
                    return

                similarity = self._cosine_similarity(openai_res, claude_res)
                
                # Calculate change rate
                prev_similarity = last_iteration["similarity"]
                change_rate = abs(similarity - prev_similarity)

                new_iteration: ModelResponse = {
                    "iteration": len(self.history) + 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                    "change_rate": change_rate,
                }
                self.history.append(new_iteration)
                
                # Check for convergence
                if similarity >= self.convergence_threshold:
                    self.converged = True
                    yield rx.toast.success(
                        f"Convergence reached! Similarity: {similarity * 100:.1f}%",
                        duration=5000
                    )

        except Exception as e:
            logging.exception(f"Error during iteration: {e}")
            yield rx.toast.error(
                f"An unexpected error occurred during iteration: {e}", duration=5000
            )
        finally:
            async with self:
                self.is_iterating = False

    @rx.event(background=True)
    async def run_automated_cycle(self):
        """Runs the automated cross-evaluation until convergence or max iterations."""
        async with self:
            if not self.history or self.automated_running:
                return
            self.automated_running = True
            self.converged = False

        oscillation_count = 0
        
        while True:
            async with self:
                if not self.automated_running:
                    break

                last_iteration = self.history[-1]
                current_iter_count = last_iteration["iteration"]
                similarity = last_iteration["similarity"]

                # Check convergence conditions
                if similarity >= self.convergence_threshold:
                    self.converged = True
                    self.automated_running = False
                    yield rx.toast.success(
                        f"✓ Converged at {similarity * 100:.1f}% similarity!",
                        duration=5000
                    )
                    break

                if current_iter_count >= self.max_iterations:
                    self.converged = False
                    self.automated_running = False
                    yield rx.toast.warning(
                        f"Max iterations ({self.max_iterations}) reached. Final similarity: {similarity * 100:.1f}%",
                        duration=5000
                    )
                    break

                # Detect oscillation (similarity not improving)
                if len(self.history) >= 3:
                    recent_changes = [h["change_rate"] for h in self.history[-3:]]
                    avg_change = sum(recent_changes) / len(recent_changes)
                    
                    if avg_change < 0.01:  # Less than 1% change on average
                        oscillation_count += 1
                        if oscillation_count >= 2:
                            self.automated_running = False
                            yield rx.toast.warning(
                                f"Responses stabilized at {similarity * 100:.1f}% similarity without reaching threshold.",
                                duration=5000
                            )
                            break
                    else:
                        oscillation_count = 0

                self._initialize_clients()
                openai_previous = last_iteration["openai_response"]
                claude_previous = last_iteration["claude_response"]

                # Generic prompt template
                new_prompt_template = """Original User Request:
{original_prompt}

Your Previous Response:
---
{your_previous_response}
---

Alternative Response from Another Model:
---
{other_model_response}
---

Instructions:
- Carefully review BOTH responses against the original user request
- Identify the strengths and weaknesses of each approach
- If the alternative response has superior reasoning, insights, or completeness, integrate those elements
- Eliminate purely stylistic differences (formatting, phrasing variations that don't affect meaning)
- Synthesize the best aspects of both responses into a refined answer
- DO NOT add meta-commentary about the comparison process
- Output ONLY your refined, complete response to the original user request"""

                openai_new_prompt = new_prompt_template.format(
                    original_prompt=self.prompt,
                    your_previous_response=openai_previous,
                    other_model_response=claude_previous,
                )
                claude_new_prompt = new_prompt_template.format(
                    original_prompt=self.prompt,
                    your_previous_response=claude_previous,
                    other_model_response=openai_previous,
                )

            try:
                openai_task = self._fetch_openai(openai_new_prompt)
                claude_task = self._fetch_claude(claude_new_prompt)
                openai_res, claude_res = await asyncio.gather(openai_task, claude_task)

                if openai_res is None or claude_res is None:
                    async with self:
                        self.automated_running = False
                    if openai_res is None:
                        yield rx.toast.error(
                            "OpenAI failed during automated cycle.", duration=5000
                        )
                    if claude_res is None:
                        yield rx.toast.error(
                            "Claude failed during automated cycle.", duration=5000
                        )
                    break

                new_similarity = self._cosine_similarity(openai_res, claude_res)
                
                # Calculate change rate
                prev_similarity = last_iteration["similarity"]
                change_rate = abs(new_similarity - prev_similarity)

                async with self:
                    new_iteration: ModelResponse = {
                        "iteration": current_iter_count + 1,
                        "openai_response": openai_res,
                        "claude_response": claude_res,
                        "similarity": new_similarity,
                        "change_rate": change_rate,
                    }
                    self.history.append(new_iteration)

            except Exception as e:
                async with self:
                    self.automated_running = False
                logging.exception(f"Error during automated iteration: {e}")
                yield rx.toast.error(
                    f"Unexpected error in automated cycle: {e}", duration=5000
                )
                break

            # Small delay between iterations to prevent rate limiting
            await asyncio.sleep(0.5)
            yield

        async with self:
            self.automated_running = False

    @rx.event
    def stop_automated_cycle(self):
        """Stops the automated evaluation cycle."""
        self.automated_running = False

    def _is_response_complete(self, response_text: str) -> bool:
        """Check if the AI response appears to be complete."""
        stripped_text = response_text.strip()
        
        # Check for common markdown code fence closure
        if "```" in stripped_text:
            return stripped_text.count("```") % 2 == 0
        
        return True

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        Returns 0.0 if either response is incomplete or invalid.
        """
        if not self._is_response_complete(text1) or not self._is_response_complete(text2):
            return 0.0

        if (
            not text1
            or not text2
            or text1.startswith("Error:")
            or text2.startswith("Error:")
        ):
            return 0.0

        # Tokenize and count word frequencies
        vec1 = Counter(text1.lower().split())
        vec2 = Counter(text2.lower().split())

        # Calculate intersection
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum((vec1[x] * vec2[x] for x in intersection))

        # Calculate magnitudes
        sum1 = sum((vec1[x] ** 2 for x in vec1.keys()))
        sum2 = sum((vec2[x] ** 2 for x in vec2.keys()))
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0

        return float(numerator) / denominator

    async def _fetch_openai(self, current_prompt: str) -> str | None:
        """Helper to fetch response from OpenAI."""
        client = cast(openai.OpenAI, self._openai_client)
        try:
            system_message = "You are a helpful assistant. Your goal is to collaborate with another AI to converge on a single, optimal response. Focus on substance and accuracy over stylistic differences."
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": current_prompt}
                ],
                max_tokens=2048,
                temperature=self.temperature,
            )
            
            response_text = (
                response.choices[0].message.content.strip() if response.choices else ""
            )
            
            if not response_text:
                logging.warning("OpenAI returned an empty response.")
                return None
            
            return response_text
            
        except Exception as e:
            logging.exception(f"OpenAI API error: {e}")
            return None

    async def _fetch_claude(self, current_prompt: str) -> str | None:
        """Helper to fetch response from Anthropic Claude."""
        client = cast(anthropic.Anthropic, self._anthropic_client)
        try:
            system_message = "You are a helpful assistant. Your goal is to collaborate with another AI to converge on a single, optimal response. Focus on substance and accuracy over stylistic differences."
            
            message = await asyncio.to_thread(
                client.messages.create,
                model=self.claude_model,
                system=system_message,
                max_tokens=2048,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=self.temperature,
            )
            
            response_text = message.content[0].text.strip() if message.content else ""
            
            if not response_text:
                logging.warning("Claude returned an empty response.")
                return None
            
            return response_text
            
        except Exception as e:
            logging.exception(f"Anthropic API error: {e}")
            return None
