import sys
import uuid
import warnings
from pathlib import Path
from typing import Annotated, ClassVar, Type, TypeVar

import tomli
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import (
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def check_path_and_handle_inheritance(path: Path, seen_files: list[Path], nested_key: str | None) -> bool | None:
    """
    Recursively look for inheritance in a toml file. Return a list of all toml files to load.

    Example:
        If config.toml has `toml_files = ["base.toml"]` and base.toml has
        `toml_files = ["common.toml"]`, this returns ["config.toml", "base.toml", "common.toml"]
        nested_key: smth like "train.optim"

    Returns:
        True if some toml inheritance is detected, False otherwise.
    """
    if path in seen_files:
        return

    if not path.exists():
        raise FileNotFoundError(f"TOML file {path} does not exist")

    with open(path, "rb") as f:
        data = tomli.load(f)

    if nested_key is not None:
        nested_keys = nested_key.split(".")
        for key in nested_keys:
            new_data = {}
            new_data[key] = data
            data = new_data

        path = get_temp_toml_file()

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    seen_files.append(path)

    recurence = False
    if "toml_files" in data:
        if nested_key is not None:
            raise NotImplementedError("--train @ helo.toml where helo.toml point to a tom file is not yet supported")
        maybe_new_files = [path.parent / file for file in data["toml_files"]]

        files = [file for file in maybe_new_files if str(file).endswith(".toml")]
        # todo which should probably look for infinite inheritance loops here
        for file in files:
            recurence = True
            check_path_and_handle_inheritance(file, seen_files, nested_key=None)

    return recurence


# Extract config file paths from CLI to pass to pydantic-settings as toml source
# This enables the use of `@` to pass config file paths to the CLI
def extract_toml_paths(args: list[str]) -> tuple[list[str], list[str]]:
    toml_paths = []
    remaining_args = args.copy()
    recurence = False
    cli_toml_file_count = 0
    for prev_arg, arg, next_arg in zip([""] + args[:-1], args, args[1:] + [""]):
        if arg == "@":
            toml_path = next_arg
            remaining_args.remove(arg)
            remaining_args.remove(next_arg)

            if prev_arg.startswith("--"):
                remaining_args.remove(prev_arg)
                nested_key = prev_arg.replace("--", "")
            else:
                nested_key = None

            recurence = recurence or check_path_and_handle_inheritance(Path(toml_path), toml_paths, nested_key)
            cli_toml_file_count += 1

    if recurence and cli_toml_file_count > 1:
        warnings.warn(
            f"{len(toml_paths)} TOML files are added via CLI ({', '.join(toml_paths)}) and at least one of them links to another file. This is not supported yet. Please either compose multiple config files via directly CLI or specify a single file linking to multiple other files"
        )

    return toml_paths, remaining_args


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

