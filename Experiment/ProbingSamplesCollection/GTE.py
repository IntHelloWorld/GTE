import os
from pathlib import Path

from GTESetup import GTESetup, ex

os.chdir(str(Path(__file__).resolve().parent))
# ex.add_config("configuration_Transformer.yaml")
# ex.add_config("configuration_Transformer_relative_pos.yaml")
# ex.add_config("configuration_Transformer_no_type.yaml")
# ex.add_config("configuration_TreeLSTM.yaml")
# ex.add_config("configuration_GRU.yaml")
ex.add_config("configuration_GGNN.yaml")


@ex.automain
def main():
    experiment = GTESetup()
    experiment.train()
