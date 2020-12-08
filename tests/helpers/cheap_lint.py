def sql_lint(sql):
    safewords = [
        # Exceptions to skip linting
    ]
    for w in safewords:
        if w in sql:
            return

    # Check whether a python string template wasn't filled
    assert "{" not in sql
    assert "}" not in sql

    # Check for balanced parentheses
    assert sql.count("(") == sql.count(")")

    # Check for balanced quote marks
    assert sql.count("'") % 2 == 0
