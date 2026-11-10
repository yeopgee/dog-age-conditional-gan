"""Convert AFAC age-in-months annotations to young/senior training labels."""

import argparse
from pathlib import Path


def age_class(months):
    if months <= 48:
        return 'Young'
    if months >= 84:
        return 'Senior'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, help='AFAC train.txt with filename and age in months.')
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    images = Path(args.images)
    rows = []
    for line in Path(args.annotations).read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        filename, age = fields
        try:
            label = age_class(int(age))
        except ValueError:
            continue
        if label and (images / filename).is_file():
            rows.append(f'{filename}\t{label}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(rows) + '\n')
    print(f'Wrote {len(rows)} labeled images to {output}')


if __name__ == '__main__':
    main()
