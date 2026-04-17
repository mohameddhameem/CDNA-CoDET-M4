"""Training utility helpers.

Notes
-----
Contains early stopping and distributed-safe checkpoint utilities.
"""

import numpy as np
import torch
import torch.distributed as dist


class EarlyStopping:
    """Track validation loss and stop training when improvement stalls."""

    def __init__(self, patience=7, verbose=False, delta=1e-7):
        """Initialize counters and thresholds for early stopping."""
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_state_dict = None

    def __call__(self, val_loss, model, path=None, is_save=True):
        """Update stopping state with the latest validation loss."""
        score = -val_loss

        if self.best_score is None:
            self._update_best(score, val_loss, model, path, is_save)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self._is_main_process() and self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self._update_best(score, val_loss, model, path, is_save)

    def _update_best(self, score, val_loss, model, path, is_save):
        """Store the new best state and optionally save a checkpoint."""
        self.best_score = score
        self._save_best_state(model)
        if is_save and path is not None:
            self._save_checkpoint(val_loss, model, path)

    def _save_best_state(self, model):
        """Cache the best model weights in CPU memory."""
        state_dict = self._get_state_dict(model)
        self.best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in state_dict.items()
        }

    def _get_state_dict(self, model):
        """Return the underlying state dict for plain or wrapped models."""
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    def _is_main_process(self):
        """Check whether the current process should perform side effects."""
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def _save_checkpoint(self, val_loss, model, path):
        """Persist the current best checkpoint to disk."""
        if not self._is_main_process():
            return

        if self.verbose:
            print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                f'Saving model ...')

        torch.save(self._get_state_dict(model), path + '/checkpoint.pth')
        self.val_loss_min = val_loss

    def load_best_model(self, model, device=None):
        """Load the cached best model weights back into the model."""
        if self.best_state_dict is None:
            return
        model_state = model.module if hasattr(model, 'module') else model
        model_state.load_state_dict(self.best_state_dict)
        if device is not None:
            model.to(device)