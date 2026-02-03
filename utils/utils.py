import ast


def parse_judge_models(model_arg):
    """Parse judge model specifications from command line argument into a dictionary."""
    judge_models = {}
    for model_spec in model_arg:
        if ":" in model_spec:
            # Format: "model:count"
            model, count = model_spec.rsplit(":", 1)
            judge_models[model] = int(count)
        else:
            # Format: "model" (defaults to 1 instance)
            judge_models[model_spec] = 1

    return judge_models


def parse_key_value_list(arg):
    """Helper function to parse a list of key-value pairs into a dictionary."""
    d = {}
    for pair in arg.split(","):
        key, value = pair.split("=", 1)
        # Try Python literal parsing (handles ints, floats, booleans, None)
        # if it fails, we'll keep it as a stirng
        # https://docs.python.org/3/library/ast.html#ast.literal_eval
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Note: not logging the error here as we are leaving the value as a string
            pass
        d[key] = value
    return d
