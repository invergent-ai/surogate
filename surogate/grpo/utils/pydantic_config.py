from pydantic import BaseModel


def to_kebab_case(args: list[str]) -> list[str]:
    """
    Converts CLI argument keys from snake case to kebab case.

    For example, `--max_batch_size 1` will be transformed `--max-batch-size 1`.
    """
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            args[i] = arg.replace("_", "-")
    return args


def get_all_fields(model: BaseModel | type) -> list[str]:
    if isinstance(model, BaseModel):
        model_cls = model.__class__
    else:
        model_cls = model

    fields = []
    for name, field in model_cls.model_fields.items():
        field_type = field.annotation
        fields.append(name)
        if field_type is not None and hasattr(field_type, "model_fields"):
            sub_fields = get_all_fields(field_type)
            fields.extend(f"{name}.{sub}" for sub in sub_fields)
    return fields


def parse_unknown_args(args: list[str], config_cls: type) -> tuple[list[str], list[str]]:
    known_fields = get_all_fields(config_cls)
    known_args = []
    unknown_args = []
    i = 0
    n = len(args)

    def get_is_key(arg: str) -> bool:
        return arg.startswith("--") or arg.startswith("-")

    while i < n:
        is_key = get_is_key(args[i])
        has_value = False if i >= n - 1 or get_is_key(args[i + 1]) else True
        if not is_key:
            i += 1
            continue
        if args[i].startswith("--"):
            key = args[i][2:]
        else:
            key = args[i][1:]
        key = key.replace("-", "_")
        if key in known_fields:
            known_args.append(args[i])
            if has_value:
                known_args.append(args[i + 1])
                i += 2
            else:
                i += 1
        else:
            unknown_args.append(args[i])
            if has_value:
                unknown_args.append(args[i + 1])
                i += 2
            else:
                i += 1

    return known_args, unknown_args

