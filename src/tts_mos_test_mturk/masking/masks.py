from typing import Tuple

import numpy as np

REVERSE_INDICATOR = "!"


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

  def reverse(self) -> None:
    self.mask = ~self.mask

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

  def clone(self) -> "MaskBase":
    raise NotImplementedError()


class RatingsMask(MaskBase):
  def clone(self) -> "RatingsMask":
    return RatingsMask(self.mask.copy())


class AssignmentsMask(MaskBase):
  def clone(self) -> "AssignmentsMask":
    return AssignmentsMask(self.mask.copy())


class WorkersMask(MaskBase):
  def clone(self) -> "WorkersMask":
    return WorkersMask(self.mask.copy())


def get_mask_name_and_reverse(mask_name: str) -> Tuple[str, bool]:
  if mask_name.startswith(REVERSE_INDICATOR):
    mask_name = mask_name[len(REVERSE_INDICATOR):]
    return mask_name, True
  return mask_name, False
