import os
from pathlib import Path

from GTESetup import GTESetup, ex

os.chdir(str(Path(__file__).resolve().parent))
ex.add_config("configuration.yaml")


@ex.automain
def main():
    experiment = GTESetup()
    experiment.train()
