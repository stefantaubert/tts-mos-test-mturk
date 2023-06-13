from argparse import ArgumentParser

from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction,
                                                    parse_non_empty_or_whitespace,
                                                    parse_output_mask_name, parse_project)


def add_req_ratings_argument(parser: ArgumentParser) -> None:
  parser.add_argument("ratings", type=parse_non_empty_or_whitespace, nargs="+", metavar="RATING",
                      help="names of ratings to use (mean of the ratings is taken)", action=ConvertToSetAction)


def add_req_project_argument(parser: ArgumentParser) -> None:
  parser.add_argument("project", type=parse_project, metavar="PROJECT-PATH",
                      help="project file (.pkl)")


def add_opt_masks_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks before operation", action=ConvertToSetAction, default=set())


def add_req_output_mask_argument(parser: ArgumentParser) -> None:
  parser.add_argument("output_mask", type=parse_output_mask_name,
                      metavar="OUTPUT-MASK", help="name of the output mask")


def add_opt_dry_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-d", "--dry", action="store_true",
                      help="dry run (i.e., don't change anything)")


# def add_encoding_argument(parser: ArgumentParser, help_str: str) -> None:
#   parser.add_argument("--encoding", type=parse_codec, metavar='CODEC',
#                       help=help_str + "; see all available codecs at https://docs.python.org/3.8/library/codecs.html#standard-encodings", default="UTF-8")
