import os
import sys
from nbconvert import PythonExporter
import nbformat
import argparse

def convert_ipynb_to_py(notebook_path, output_path=None):
    """
    Convert a single .ipynb file to a .py file.

    Args:
        notebook_path (str): Path to the .ipynb file.
        output_path (str, optional): Path to save the .py file. If None, saves in the same directory.
    """
    try:
        # Read the notebook file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Initialize the exporter
        exporter = PythonExporter()

        # Convert the notebook to Python script
        script, _ = exporter.from_notebook_node(notebook)

        # Determine the output path
        if output_path is None:
            base, _ = os.path.splitext(notebook_path)
            output_path = base + '.py'

        # Write the Python script
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)

        print(f"Converted: {notebook_path} -> {output_path}")

    except Exception as e:
        print(f"Failed to convert {notebook_path}: {e}")

def traverse_and_convert(root_dir, replace=False):
    """
    Traverse the directory tree and convert all .ipynb files to .py.

    Args:
        root_dir (str): The root directory to start traversal.
        replace (bool): If True, delete the original .ipynb file after conversion.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ipynb'):
                notebook_path = os.path.join(dirpath, filename)
                convert_ipynb_to_py(notebook_path)

                if replace:
                    try:
                        os.remove(notebook_path)
                        print(f"Removed original notebook: {notebook_path}")
                    except Exception as e:
                        print(f"Failed to remove {notebook_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert all Jupyter .ipynb files to .py scripts."
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Root directory to start conversion (default: current directory)'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace .ipynb files by deleting them after conversion'
    )

    args = parser.parse_args()
    root_dir = os.path.abspath(args.directory)

    if not os.path.isdir(root_dir):
        print(f"Error: The directory '{root_dir}' does not exist.")
        sys.exit(1)

    traverse_and_convert(root_dir, replace=args.replace)
    print("Conversion process completed.")

if __name__ == "__main__":
    main()

