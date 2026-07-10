#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import argparse
import os
import re


def parse_requirements(file_path):
    """
    Parses a requirements file into a dictionary of package name -> full requirement string.
    """
    requirements = {}
    if not os.path.exists(file_path):
        return requirements

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Simple regex to extract the package name.
            # This handles 'package', 'package==1.0', 'package>=1.0', etc.
            # It also handles git links if they have a #egg=package suffix.
            match = re.match(r"^([^<>=! ]+)", line)
            if match:
                package_name = match.group(1).lower()
                requirements[package_name] = line
            else:
                # If we can't parse it, just store it with line as key to preserve it
                requirements[line] = line
    return requirements


def merge_requirements(base_file, override_file, output_file=None):
    """
    Merges override_file into base_file.
    Duplicate dependencies from override_file take precedence.
    """
    base_reqs = parse_requirements(base_file)
    override_reqs = parse_requirements(override_file)

    # In Python 3.9+, we could use base_reqs | override_reqs
    # But for compatibility with older Python 3 versions:
    merged = base_reqs.copy()
    merged.update(override_reqs)

    content = "\n".join(merged.values()) + "\n"

    if output_file:
        with open(output_file, "w") as f:
            f.write(content)
        print(f"Merged requirements saved to {output_file}")
    else:
        print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two requirements.txt files. Secondary file overrides the primary."
    )
    parser.add_argument("base", help="The primary requirements file")
    parser.add_argument("override", help="The requirements file that overrides the primary")
    parser.add_argument("-o", "--output", help="Output file path (prints to stdout if not provided)")

    args = parser.parse_args()
    merge_requirements(args.base, args.override, args.output)
