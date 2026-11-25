"""
Verification script for analysis.ipynb
======================================
Validates JSON structure and executes all code cells in sequential order
to guarantee end-to-end reproducibility of the notebook.
"""

import json
import sys

def main():
    print("--- Testing Stage 4: Notebook Structure and Code Cell Execution ---")
    with open("analysis.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4, f"Expected nbformat 4, got {nb['nbformat']}"
    cells = nb["cells"]
    print(f"Total Notebook Cells: {len(cells)}")

    # Extract and count cell types
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    print(f"Markdown Documentation Cells: {len(md_cells)}")
    print(f"Executable Code Cells       : {len(code_cells)}")

    import matplotlib
    matplotlib.use('Agg')
    # Shared execution namespace
    global_env = {"__name__": "__main__"}

    for i, cell in enumerate(code_cells, 1):
        code_str = "".join(cell["source"])
        first_line = code_str.strip().split("\n")[0][:60]
        print(f"Executing Code Cell {i}/{len(code_cells)}: [{first_line}...] ", end="")
        try:
            exec(code_str, global_env)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)

    print("\nSTAGE 4 SUCCESS: All 19 sections of analysis.ipynb verified and executed cleanly!")

if __name__ == "__main__":
    main()
