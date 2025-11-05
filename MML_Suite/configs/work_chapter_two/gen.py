from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate IEMOCAP dataset configuration")
    parser.add_argument("--template", type=str, required=True,)


    args = parser.parse_args()
    template = args.template
    print(f"Template: {template}")

    template_str = open(template, "r").read()
    for i in range(1, 13):
        formatted = template_str.replace(r"${CV_NO}", str(i))
        with open(f"{template.rsplit('_', 1)[0]}_{i}.yaml", "w") as f:
            f.write(formatted)
            fil = f"{template.rsplit('_', 1)[0]}_{i}.yaml"
            print(f"Wrote configuration for CV {i} to {fil}")
    print("All configurations generated successfully.")
    