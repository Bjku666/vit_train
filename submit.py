"""Stable entry-point for generating submission CSV.

We keep all real logic in `inference.py`.
Shell scripts call `python submit.py ...` so you can rename/refactor
`inference.py` without breaking your workflow.
"""

from inference import main


if __name__ == "__main__":
    main()
