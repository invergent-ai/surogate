def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj

def get_not_null(value, default=None):
    """
    return the value if it's not None, otherwise return the default value
    """
    return value if value is not None else default

