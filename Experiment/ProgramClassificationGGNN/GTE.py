from GTESetup import GTESetup, ex

ex.add_config("Experiment/ProgramClassificationGGNN/configuration.yaml")


@ex.automain
def main():
    experiment = GTESetup()
    experiment.train()
