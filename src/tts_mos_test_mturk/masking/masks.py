import numpy as np


class MaskBase():
  def __init__(self, mask: np.ndarray) -> None:
    self.mask = mask

  def combine_mask(self, mask: "MaskBase") -> None:
    self.combine_mask_np(mask.mask)

  def combine_mask_np(self, mask: np.ndarray) -> None:
    assert mask.shape == self.mask.shape
    self.mask = (self.mask | mask)

  def mask_indices(self, indices: np.ndarray) -> None:
    self.mask[indices] = True

  def apply_by_nan(self, data: np.ndarray) -> None:
    assert data.shape == self.mask.shape
    data[self.mask] = np.nan

  def apply_by_false(self, data: np.ndarray) -> None:
    assert data.shape == self.mask.shape
    data[self.mask] = False

  def apply_by_del(self, data: np.ndarray) -> np.ndarray:
    assert data.shape == self.mask.shape
    data = data[~self.mask]
    return data

  @property
  def n_masked(self) -> int:
    result = np.sum(self.mask)
    return result

  @property
  def n_unmasked(self) -> int:
    result = np.sum(~self.mask)
    return result

  @property
  def masked_indices(self) -> np.ndarray:
    result = self.mask.nonzero()[0]
    return result

  @property
  def unmasked_indices(self) -> np.ndarray:
    result = (~self.mask).nonzero()[0]
    return result


class RatingsMask(MaskBase):
  pass


class AssignmentsMask(MaskBase):
  pass


class WorkersMask(MaskBase):
  pass
