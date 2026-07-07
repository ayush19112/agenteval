import functools


def with_example(fn, **example_args):
    """A copy of a tool carrying different example args.

    Example args only steer the offline mock model (they travel in the tool
    schema as `x_example_args`); real models choose their own arguments.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.example_args = example_args
    return wrapper
