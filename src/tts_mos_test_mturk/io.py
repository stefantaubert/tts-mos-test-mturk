import pickle
from pathlib import Path
from typing import Any


def save_obj(obj: Any, path: Path) -> None:
  assert isinstance(path, Path)
  # assert path.parent.exists() and path.parent.is_dir()
  with open(path, mode="wb") as file:
    pickle.dump(obj, file)


def load_obj(path: Path) -> Any:
  assert isinstance(path, Path)
  # assert path.is_file()
  with open(path, mode="rb") as file:
    return pickle.load(file)
