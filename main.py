import asyncio
import difflib
import json
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
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.widgets import Button, Dialog, RadioList
from pygments.lexers.python import PythonLexer

from tokyonight_pygments import tokyonight_style

# -----------------------------
# Constants
# -----------------------------
with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)
ALL_INSTRUCTIONS = (
    "Your response should be a string of Python code (without markdown formatting or any explanation) "
    "that replaces the original code as appropriate.\n\nCODE:\n\n"
)


# -----------------------------
# Model Setup
# -----------------------------
def setup_model_client() -> list[str]:
    """
    Sets up the model client for LLM operations.

    This function attempts to connect to the LM Studio server and retrieve
    a list of loaded models. If connection fails, it returns an empty list
    and prints an error message.

    Returns:
        list[str]: A list of loaded model identifiers. Returns empty list if connection fails.
    """
    models_available: list[str] = []

    try:
        with lms.Client() as init_client:
            downloaded = [llm.identifier for llm in init_client.llm.list_loaded()]
        client = lms.get_default_client()
        models_available = downloaded
    except Exception as e:
        print(
            f"Could not connect to LM Studio server. Please make sure it's running. Error: {e}"
        )

    return models_available


models_available = setup_model_client()


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
    spin = f" [{spinner_text[0]}]" if spinner_running[0] else ""
    return f" Model: {model} | Prompt: {prompt_name}{flash}{spin} "


active_selector = [None]  # will hold the RadioList widget when open
active_accept = [None]  # will hold a callable to commit selection


def close_popups(app, restore_focus: bool = True):
    active_selector[0] = None
    active_accept[0] = None
    root_container.floats[:] = []
    if restore_focus:
        app.layout.focus(code_control)
    app.invalidate()


def _make_selector_dialog(title: str, items, initial_value, on_accept):
    selector = RadioList([(x, x) for x in items])
    if initial_value in items:
        selector.current_value = initial_value

    def ok_handler():
        if selector.current_value is not None:
            on_accept(selector.current_value)
        close_popups(app)  # always closes and restores focus

    def cancel_handler():
        close_popups(app)

    dialog = Dialog(
        title=title,
        body=selector,
        buttons=[
            Button(text="OK", handler=ok_handler),
            Button(text="Cancel", handler=cancel_handler),
        ],
        width=None,
        modal=True,
    )
    return selector, dialog


def show_model_popup(app):
    if not models_available:
        output_buffer.set_document(
            Document("No models loaded. Start LM Studio and load a model."),
            bypass_readonly=True,
        )
        return
    close_popups(app)
    selector, dialog = _make_selector_dialog(
        "Select Model",
        models_available,
        current_model[0],
        on_accept=lambda val: current_model.__setitem__(0, val),
    )
    active_selector[0] = selector
    active_accept[0] = lambda: (
        current_model.__setitem__(0, selector.current_value),
        None,
    )
    root_container.floats.append(Float(content=dialog))
    app.layout.focus(selector)


def show_prompt_popup(app):
    close_popups(app)
    prompt_names = list(PROMPTS.keys())
    selector, dialog = _make_selector_dialog(
        "Select Prompt",
        prompt_names,
        current_prompt[0],
        on_accept=lambda val: current_prompt.__setitem__(0, val),
    )
    active_selector[0] = selector
    active_accept[0] = lambda: (
        current_prompt.__setitem__(0, selector.current_value),
        None,
    )
    root_container.floats.append(Float(content=dialog))
    app.layout.focus(selector)


status_bar = Window(
    height=1,
    content=FormattedTextControl(lambda: status_text()),
    style="bg:#c099ff fg:#222436",
)

# -----------------------------
# Layout (fixed 50/50 scrollable)
# -----------------------------
input_pane = HSplit(
    [
        Window(
            height=1,
            content=FormattedTextControl("Input:"),
            style="bg:#82aaff fg:#222436",
        ),
        Window(code_control, wrap_lines=True, always_hide_cursor=False),
    ],
    height=D(weight=1),
)

output_pane = HSplit(
    [
        Window(
            height=1,
            content=FormattedTextControl("Output:"),
            style="bg:#82aaff fg:#222436",
        ),
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


@kb.add("L")
def _(event):
    # Open model popup; if one is already open, close and replace.
    show_model_popup(event.app)


@kb.add("P")
def _(event):
    # Open prompt popup; if one is already open, close and replace.
    show_prompt_popup(event.app)


@kb.add("N")
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
    style=tokyonight_style,
)

if __name__ == "__main__":
    app.run()
