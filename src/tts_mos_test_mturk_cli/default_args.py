from argparse import ArgumentParser
from multiprocessing import cpu_count

from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, ConvertToSetAction,
                                                    get_optional, parse_codec, parse_existing_file,
                                                    parse_non_empty_or_whitespace,
                                                    parse_positive_integer)

DEFAULT_N_JOBS = cpu_count()
DEFAULT_CHUNKSIZE = 1_000_000
DEFAULT_MAXTASKSPERCHILD = None


def add_from_and_to_subsets_arguments(parser: ArgumentParser) -> None:
  add_from_subsets_argument(parser)
  add_to_subset_argument(parser)


def add_project_argument(parser: ArgumentParser) -> None:
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")


def add_masks_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks before operation", action=ConvertToSetAction, default=set())


def add_output_mask_argument(parser: ArgumentParser) -> None:
  parser.add_argument("output_mask", type=parse_non_empty_or_whitespace,
                      metavar="OUTPUT-MASK", help="name of the output mask")


def add_from_subsets_argument(parser: ArgumentParser) -> None:
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="FROM-SUBSET", help="from subset", action=ConvertToOrderedSetAction)


def add_to_subset_argument(parser: ArgumentParser) -> None:
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace,
                      metavar="TO-SUBSET", help="to subset")


def add_dry_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-d", "--dry", action="store_true",
                      help="dry run (i.e., don't change anything)")


def add_dataset_argument(parser: ArgumentParser) -> None:
  parser.add_argument("dataset", type=parse_existing_file, metavar="DATASET-PATH",
                      help="dataset file")


def add_sep_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--sep", type=str, metavar="STRING",
                      help="separator for units", default="")


def add_mp_group(parser: ArgumentParser):
  group = parser.add_argument_group("multiprocessing arguments")
  add_n_jobs_argument(group)
  add_chunksize_argument(group)
  add_maxtasksperchild_argument(group)
  return group


def add_n_jobs_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-j", "--n-jobs", metavar='N', type=int,
                      choices=range(1, cpu_count() + 1), default=DEFAULT_N_JOBS, help="amount of parallel cpu jobs")


def add_chunksize_argument(parser: ArgumentParser, target: str = "files", default: int = DEFAULT_CHUNKSIZE) -> None:
  parser.add_argument("-s", "--chunksize", type=parse_positive_integer, metavar="SIZE",
                      help=f"amount of {target} to chunk into one job", default=default)


def add_maxtasksperchild_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--maxtasksperchild", type=get_optional(parse_positive_integer), metavar="COUNT",
                      help="amount of tasks per child", default=DEFAULT_MAXTASKSPERCHILD)


def add_encoding_argument(parser: ArgumentParser, help_str: str) -> None:
  parser.add_argument("--encoding", type=parse_codec, metavar='CODEC',
                      help=help_str + "; see all available codecs at https://docs.python.org/3.8/library/codecs.html#standard-encodings", default="UTF-8")
