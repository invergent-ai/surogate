"""Legacy artifacts must be rebuilt from their checkpoint's resolved configuration."""


def main(argv=None):
    raise SystemExit(
        "Legacy artifacts do not declare the checkpoint dimensions and execution settings "
        "required by serving. Rebuild the serving artifact from its source checkpoint."
    )


if __name__ == "__main__":
    main()
