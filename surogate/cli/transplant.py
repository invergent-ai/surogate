import argparse
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--student",
        help="Student model directory or HuggingFace ID (the model to be trained). Required unless --restore.",
    )
    parser.add_argument(
        "--teacher",
        help="Teacher model directory or HuggingFace ID (tokenizer donor). Required unless --restore.",
    )
    parser.add_argument("--output", required=True, help="Output directory for the transplanted model")
    parser.add_argument(
        "--method",
        default="omp",
        help="Approximation method for new vocabulary rows (default: omp, recommended)",
    )
    parser.add_argument(
        "--k", type=int, default=64, help="Sparsity level / neighbor count for the approximation (default: 64)"
    )
    parser.add_argument("--device", default=None, help="Device for the approximation solve (e.g. cuda)")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Pass --trust-remote-code to mergekit-tokensurgeon"
    )
    parser.add_argument(
        "--restore",
        metavar="MANIFEST",
        default=None,
        help="Reverse mode: path to a transplant_manifest.json; transplants the distilled model "
        "(given via --student) back to the original student tokenizer.",
    )

    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.transplant import run_transplant, run_transplant_back

    if args.restore:
        if not args.student:
            logger.error("--restore requires --student (the KD-trained model to convert back).")
            sys.exit(1)
        run_transplant_back(
            distilled_model=args.student,
            manifest_path=args.restore,
            out_dir=args.output,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        if not args.student or not args.teacher:
            logger.error("transplant-tokenizer requires --student and --teacher (or --restore MANIFEST).")
            sys.exit(1)
        run_transplant(
            student_model=args.student,
            teacher_model=args.teacher,
            out_dir=args.output,
            method=args.method,
            k=args.k,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
