import asyncio
import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import lmstudio as lms
from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import (
    FloatContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.lexers.python import PythonLexer
from pygments.styles import get_style_by_name

# -----------------------------
# Constants
# -----------------------------
PROMPTS = {
    "Add type hints & documentation": """Modify the following Python code by adding appropriate type hints and comprehensive docstrings. Make sure the docstrings follow Google style guidelines with Args, Returns, and Raises sections where applicable. Return the full code with updates applied. Do NOT wrap the code in a markdown block (such as ```python, etc) - only write pure Python code.""",
    "Create unit tests for this function": """Generate comprehensive unit tests for the following Python code using pytest. Include tests for normal cases, edge cases, and error conditions where appropriate.""",
    "Refactor for readability & speed": """Refactor the following Python code to improve both readability and performance. Focus on simplifying logic, using more efficient data structures where possible, and making the code more maintainable.""",
    "Identify possible bugs": """Analyze the following Python code and identify any potential bugs, edge cases, or areas for improvement. Provide specific suggestions for fixes or enhancements.""",
}

ALL_INSTRUCTIONS = (
    "Your response should be a string of Python code (without markdown formatting or any explanation) "
    "that replaces the original code as appropriate.\n\nCODE:\n\n"
)

style = style_from_pygments_cls(get_style_by_name("monokai"))

# -----------------------------
# Model Setup
# -----------------------------
try:
    with lms.Client() as init_client:
        downloaded = [llm.identifier for llm in init_client.llm.list_loaded()]
    client = lms.get_default_client()
    models_available = downloaded
except Exception as e:
    models_available = []
    print(
        f"Could not connect to LM Studio server. Please make sure it's running. Error: {e}"
    )


# -----------------------------
# Helpers
# -----------------------------


def apply_ruff_fixes(code: str) -> str:
    """Format and autofix Python code with ruff, returning the cleaned version.

    This function takes a string of Python code and applies ruff formatting and
    autofixes to it. It uses a temporary file to process the code and returns
    the cleaned version.

    Args:
        code: A string containing Python code to be formatted and fixed.

    Returns:
        A string containing the formatted and autofixed Python code.

    Raises:
        subprocess.SubprocessError: If ruff is not installed or cannot be executed.
        OSError: If temporary file operations fail.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "snippet.py"
        tmp_path.write_text(code)

        # Apply autofixes and formatting
        subprocess.run(
            ["ruff", "check", "--fix", str(tmp_path)],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["ruff", "format", str(tmp_path)],
            check=False,
            capture_output=True,
        )

        return tmp_path.read_text()


def run_model(client, llm_model: str, full_prompt: str) -> str:
    try:
        model = client.llm.model(llm_model)
        return model.respond(full_prompt).content
    except Exception as e:
        return f"An error occurred: {e}"


def clean_code_response(text: str) -> str:
    fence_re = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", re.MULTILINE)
    m = fence_re.search(text.strip())
    return m.group(1).strip() if m else text.strip()


def make_diff(original: str, modified: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile="input.py",
        tofile="output.py",
        lineterm="",
    )
    return "\n".join(diff)


# -----------------------------
# Buffers & State
# -----------------------------
code_buffer = Buffer()
output_buffer = Buffer(read_only=True)

code_control = BufferControl(
    buffer=code_buffer,
    lexer=PygmentsLexer(PythonLexer),
    focusable=True,
)
output_control = BufferControl(
    buffer=output_buffer,
    lexer=PygmentsLexer(PythonLexer),
    focusable=True,
)

current_model = [models_available[0] if models_available else None]
current_prompt = [next(iter(PROMPTS))]

showing_diff = [False]
flash_active = [False]
last_response = [""]  # always keep the last pure LLM output

clipboard = PyperclipClipboard()

spinner_running = [False]
spinner_text = [""]


async def spinner_task():
    frames = ["|", "/", "-", "\\"]
    i = 0
    spinner_running[0] = True
    while spinner_running[0]:
        spinner_text[0] = frames[i % len(frames)]
        get_app().invalidate()
        await asyncio.sleep(0.1)
        i += 1
    spinner_text[0] = ""
    get_app().invalidate()


def status_text():
    model = current_model[0] or "None"
    prompt_name = current_prompt[0]
    flash = " (YANKED!)" if flash_active[0] else ""
    spin = f" [Running {spinner_text[0]}]" if spinner_running[0] else ""
    return f" Model: {model} | Prompt: {prompt_name}{flash}{spin} "


status_bar = Window(
    height=1,
    content=FormattedTextControl(lambda: status_text()),
    style="reverse",
)

# -----------------------------
# Layout (fixed 50/50 scrollable)
# -----------------------------
input_pane = HSplit(
    [
        Window(height=1, content=FormattedTextControl("Input:"), style="reverse"),
        Window(code_control, wrap_lines=True, always_hide_cursor=False),
    ],
    height=D(weight=1),
)

output_pane = HSplit(
    [
        Window(height=1, content=FormattedTextControl("Output:"), style="reverse"),
        Window(output_control, wrap_lines=True, always_hide_cursor=False),
    ],
    height=D(weight=1),
)

body = HSplit([input_pane, output_pane, status_bar])

root_container = FloatContainer(content=body, floats=[])
layout = Layout(root_container, focused_element=code_control)

# -----------------------------
# Keybindings
# -----------------------------
kb = KeyBindings()


@kb.add("tab")
def _(event):
    event.app.layout.focus_next()


@kb.add("s-tab")
def _(event):
    event.app.layout.focus_previous()


@kb.add("c-n")
def _(event):
    # Reset state
    code_buffer.text = ""
    output_buffer.set_document(Document(""), bypass_readonly=True)
    last_response[0] = ""
    showing_diff[0] = False
    flash_active[0] = False
    spinner_running[0] = False
    spinner_text[0] = ""
    event.app.layout.focus(code_control)
    event.app.invalidate()


@kb.add("c-q")
def _(event):
    event.app.exit()


@kb.add("c-r")
def _(event):
    llm_model = current_model[0]
    selected_prompt = current_prompt[0]
    code_text = code_buffer.text

    if not llm_model or not selected_prompt or not code_text.strip():
        output_buffer.set_document(
            Document("Please provide code, model, and prompt."), bypass_readonly=True
        )
        return

    full_prompt = PROMPTS[selected_prompt] + "\n\n" + ALL_INSTRUCTIONS + code_text

    async def run_and_update():
        # start spinner
        task = get_app().create_background_task(spinner_task())
        try:
            response = await asyncio.to_thread(
                run_model, client, llm_model, full_prompt
            )
            cleaned = clean_code_response(response)
            fixed = apply_ruff_fixes(cleaned)
            last_response[0] = fixed
            output_buffer.set_document(Document(fixed), bypass_readonly=True)
        finally:
            spinner_running[0] = False

    event.app.create_background_task(run_and_update())


@kb.add("c-d")
def _(event):
    showing_diff[0] = not showing_diff[0]
    if showing_diff[0]:
        diffed = make_diff(code_buffer.text, last_response[0])
        output_buffer.set_document(Document(diffed), bypass_readonly=True)
    else:
        output_buffer.set_document(Document(last_response[0]), bypass_readonly=True)


@kb.add("c-y")
def _(event):
    clipboard.set_text(output_buffer.text)
    flash_active[0] = True
    event.app.invalidate()

    def clear_flash():
        flash_active[0] = False
        get_app().invalidate()

    event.app.create_background_task(_flash_reset(clear_flash))


async def _flash_reset(cb):
    await asyncio.sleep(0.5)
    cb()


# Scroll bindings (for current window)
@kb.add("c-up")
def _(event):
    w = event.app.layout.current_window
    if w:
        w.vertical_scroll -= 1


@kb.add("c-down")
def _(event):
    w = event.app.layout.current_window
    if w:
        w.vertical_scroll += 1


# -----------------------------
# Application
# -----------------------------
app = Application(
    layout=layout,
    full_screen=True,
    key_bindings=kb,
    style=style,
)

if __name__ == "__main__":
    app.run()
