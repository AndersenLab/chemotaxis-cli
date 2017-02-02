import sys
import argparse
from . import _program, crop_and_filter_plate
from clint.textui import puts, indent, colored
import os

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main(args = sys.argv[1:]):
    parser = argparse.ArgumentParser(prog = _program)

    parser.add_argument('images', type=argparse.FileType('r'), nargs='+')

    parser.add_argument("--radius",
                        help="Set the radius (px) of the plate",
                        type=int,
                        default=930)

    parser.add_argument("-e",
                        "--crop",
                        help="Additional crop from edge",
                        type=int,
                        default=100)

    parser.add_argument("-p",
                        "--particle-size",
                        help="Filter out particles smaller than specified",
                        type=int,
                        default=100)

    parser.add_argument("-d",
                        "--debug",
                        help="Output debug information",
                        action="store_true",
                        default=False)

    args = parser.parse_args(args)

    radius_range = [args.radius - 20, args.radius + 20]

    if args.debug:
        make_dir("debug")

    for img in args.images:
        with indent(4):
            puts(colored.blue("\nProcessing " + img.name))
            img = crop_and_filter_plate(img.name,
                                        radius_range,
                                        extra_crop=args.crop,
                                        debug=args.debug)


if __name__ == '__main__':
    main()
