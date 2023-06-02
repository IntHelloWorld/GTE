import os
from pathlib import Path

os.chdir(Path(__file__).parent)

from GTESetup import GTESetup, ex

ex.add_config("configuration.yaml")


@ex.automain
def main():
    experiment = GTESetup()
    experiment.train()
