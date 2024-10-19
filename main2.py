def remove_comments(code: str) -> str:
    """
    Removes single-line and multi-line comments from Python code.
    """
    import io
    import tokenize

    try:
        io_obj = io.StringIO(code)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            ltext = tok.line

            if token_type == tokenize.COMMENT:
                continue
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # Likely a docstring
                    if start_col == 0:
                        continue
            out += token_string
            prev_toktype = token_type
        return out
    except Exception as e:
        logging.warning(f"Failed to remove comments: {e}")
        return code  # Return original code if something goes wrong
