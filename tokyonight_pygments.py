from prompt_toolkit.styles import Style

tokyonight_style = Style.from_dict(
    {
        # Basic text colors
        "pygments.text": "#c8d3f5",
        "pygments.keyword": "#82aaff",
        "pygments.keyword.constant": "#ff966c",
        "pygments.keyword.declaration": "#82aaff",
        "pygments.keyword.namespace": "#82aaff",
        "pygments.keyword.pseudo": "#82aaff",
        "pygments.keyword.reserved": "#82aaff",
        "pygments.keyword.special": "#82aaff",
        # Strings and literals
        "pygments.literal.string": "#c3e88d",
        "pygments.literal.string.doc": "#636da6",
        "pygments.literal.string.escape": "#ff966c",
        "pygments.literal.string.interpol": "#ffc777",
        "pygments.literal.string.other": "#c3e88d",
        "pygments.literal.string.regex": "#ff966c",
        "pygments.literal.string.single": "#c3e88d",
        "pygments.literal.string.double": "#c3e88d",
        "pygments.literal.string.backtick": "#c3e88d",
        # Numbers and constants
        "pygments.literal.number": "#ff966c",
        "pygments.literal.number.integer": "#ff966c",
        "pygments.literal.number.float": "#ff966c",
        "pygments.literal.number.hex": "#ff966c",
        "pygments.literal.number.oct": "#ff966c",
        "pygments.literal.number.bin": "#ff966c",
        # Comments
        "pygments.comment": "#636da6",
        "pygments.comment.multiline": "#636da6",
        "pygments.comment.single": "#636da6",
        "pygments.comment.preproc": "#636da6",
        "pygments.comment.special": "#636da6",
        # Functions and methods
        "pygments.name.function": "#82aaff",
        "pygments.name.function.magic": "#82aaff",
        "pygments.name.class": "#82aaff",
        "pygments.name.namespace": "#82aaff",
        "pygments.name.exception": "#ff757f",
        "pygments.name.variable": "#c8d3f5",
        "pygments.name.variable.magic": "#c8d3f5",
        # Operators and punctuation
        "pygments.operator": "#82aaff",
        "pygments.operator.word": "#82aaff",
        "pygments.punctuation": "#82aaff",
        # Types and classes
        "pygments.literal.type": "#ff966c",
        "pygments.name.builtin": "#ff966c",
        # Built-in constants
        "pygments.name.constant": "#ff966c",
        "pygments.name.variable.global": "#c8d3f5",
        # Keywords with special meanings
        "pygments.keyword.control": "#82aaff",
        "pygments.keyword.operator": "#82aaff",
        "pygments.keyword.other": "#82aaff",
        # Special tokens
        "pygments.generic.heading": "#c3e88d",
        "pygments.generic.subheading": "#ff966c",
        "pygments.generic.deleted": "#ff757f",
        "pygments.generic.inserted": "#c3e88d",
        "pygments.generic.changed": "#ffc777",
        # Background colors
        "pygments.background": "#222436",
        "pygments.background.dark": "#1e2030",
        # Highlight colors
        "pygments.highlight": "#2f334d",
        # Other common tokens
        "pygments.name": "#c8d3f5",
        "pygments.error": "#ff757f",
        "pygments.warning": "#ffc777",
        # For prompt_toolkit specific elements
        "toolbar": "bg:#222436 #c8d3f5",
        "toolbar.border": "#636da6",
        "frame.border": "#2f334d",
        "frame.title": "#82aaff",
        # Syntax highlighting for different languages
        "pygments.keyword.type": "#ff966c",
        "pygments.keyword.argument": "#82aaff",
        # Special syntax elements
        "pygments.name.tag": "#ff966c",
        "pygments.name.attribute": "#ffc777",
        "pygments.name.label": "#82aaff",
    }
)

# Alternative more minimal version:
minimal_style = Style.from_dict(
    {
        # Basic syntax highlighting
        "pygments.keyword": "#82aaff",
        "pygments.literal.string": "#c3e88d",
        "pygments.literal.number": "#ff966c",
        "pygments.comment": "#636da6",
        "pygments.name.function": "#82aaff",
        "pygments.name.class": "#82aaff",
        "pygments.name.variable": "#c8d3f5",
        "pygments.operator": "#82aaff",
        # Background and text
        "pygments.background": "#222436",
        "pygments.text": "#c8d3f5",
        # Special cases
        "pygments.name.constant": "#ff966c",
        "pygments.name.builtin": "#ff966c",
    }
)
