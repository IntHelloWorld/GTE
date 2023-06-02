from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, run_id, patience, logger):
        self.run_id = run_id
        self.patience = patience
        self.logger = logger
        self.checkpoint = False
        self._counter = 0
        self._best = 0
        self.model_name = None

    def evaluate(self, score):
        if score > self._best or self._best == 0:
            self.logger.info(f"Score improved: ({self._best:.4f} -> {score:.4f})")
            self._best = score
            self._counter = 0
            self.checkpoint = True
        else:
            self._counter += 1
            self.checkpoint = False

        self.logger.info(f"Early stop counter: {self._counter}, Patience: {self.patience}, Current score: {score:.4f}, Best score: {self._best:.4f}")

        if self._counter > self.patience:
            return False
        else:
            return True

    def save_checkpoint(self, output_dir, file_name, model):
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)

        if self.model_name is not None:
            Path(self.model_name).unlink()
        self.model_name = str(Path(output_dir) / file_name)
        torch.save(model.state_dict(), self.model_name)
        self.logger.info(f"Model saved to {self.model_name}")

    @property
    def best(self):
        return self._best

    @property
    def counter(self):
        return self._counter
