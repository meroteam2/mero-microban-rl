"""`microban-play`: MJLab's play CLI with the Microban task registered."""

import microban_rl  # noqa: F401  (registers Microban-Stand)
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
