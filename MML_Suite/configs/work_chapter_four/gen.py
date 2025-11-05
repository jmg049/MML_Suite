from argparse import ArgumentParser
from pathlib import Path
import re

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Cross-Validation Configs for Chapter Four")
    parser.add_argument(
        "-n",
        "--num_folds",
        type=int,
        help="Number of folds for cross-validation",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory to save the generated configs",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--template_file",
        type=str,
        help="Template file to use for generating configs",
        required=True,
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.template_file).parent

    else:
        args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_file = Path(args.template_file)
    with open(template_file, "r") as file:
        template = file.read()
    for fold in range(args.num_folds):
        # config = template.replace("{{CV_NO}}", str(fold + 1))

        config = re.sub(
            r"{CV_NO}", str(fold + 1), template
        )
        debug_matches = re.findall(r"{CV_NO}", template)
        if len(debug_matches) == 0:
            print("No {{CV_NO}} found in the template file. Please check the template.")
        elif len(debug_matches) > 1:
            print(f"Found {len(debug_matches)} occurrences of {{CV_NO}} in the template file. Replacing all.")

        # and does not overwrite the template file        

        config_file = args.output_dir / template_file.name.replace(
            "_template", f"_fold_{fold + 1}.yaml"
        )
        print(f"Writing config for fold {fold + 1} to {config_file}")
        assert config_file != template_file, "Config file must not be the same as the template file"
        with open(config_file, "w") as file:
            file.write(config)
    print(f"All configs generated in {args.output_dir}")
    print(f"Template file used: {template_file}")
    print(f"Number of folds: {args.num_folds}")
    print(f"Output directory: {args.output_dir}")
