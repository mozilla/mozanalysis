import re


def rstify_links(s):
    return re.sub(r"\[(.*?)\]\((.*?)\)", r"`\1 <\2>`_", s)


# """unit tests"""
assert rstify_links("[foo](https://foo.bar)") == "`foo <https://foo.bar>`_"
assert (
    rstify_links("[foo bar](https://example.com?frob) baz")
    == "`foo bar <https://example.com?frob>`_ baz"
)


def process(app, what, name, obj, options, lines):
    if not hasattr(obj, "description"):
        return
    for i, line in enumerate(lines):
        if "..." in line:
            lines[i] = line.replace("...", rstify_links(obj.description))


def setup(app):
    app.connect("autodoc-process-docstring", process)
    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
