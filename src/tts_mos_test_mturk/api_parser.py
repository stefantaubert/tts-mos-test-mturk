

import re
from pathlib import Path
from typing import Dict, List

import boto3
import botocore
import pandas as pd
import xmltodict
from mypy_boto3_mturk.type_defs import HITTypeDef

MOS_PATTERN = re.compile(r"(\d+)-mos-rating\.([1-5])")
LT_PATTERN = re.compile(r"listening-type\.(.+)")


def parse_mos_answers(res: Dict[str, str]) -> Dict[str, int]:
  result = {}
  for identifier, val in res.items():
    if val == "true":
      mos_match = re.match(MOS_PATTERN, identifier)
      if isinstance(mos_match, re.Match):
        sample_nr, mos_val = mos_match.groups()
        assert sample_nr not in result
        result[sample_nr] = int(mos_val)
  return result


def parse_listening_type(res: Dict[str, str]) -> str:
  result = None
  for identifier, val in res.items():
    if val == "true":
      mos_match = re.match(LT_PATTERN, identifier)
      if isinstance(mos_match, re.Match):
        lt = mos_match.group(1)
        assert result is None
        result = lt
  return result


def parse_comment(res: Dict[str, str]) -> str:
  result = res.get("comment", "")
  return result


def answer_to_dict(answer: str) -> Dict[str, str]:
  xml_doc = xmltodict.parse(answer)
  tmp = {
    answer_field['QuestionIdentifier']: answer_field['FreeText']
    for answer_field in xml_doc['QuestionFormAnswers']['Answer']
  }
  return tmp


def parse_api(key_id: str, access_key: str):

  MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
  MTURK_PRODUCTIVE = None

  session = boto3.Session(
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    region_name="us-east-1",
  )

  MTURK_ENDPOINT = MTURK_SANDBOX

  mturk = session.client('mturk', endpoint_url=MTURK_ENDPOINT)
  x = mturk.list_hits()

  # hit: HITTypeDef
  for hit in x["HITs"]:
    worker_results = mturk.list_assignments_for_hit(HITId=hit["HITId"])
    # h = mturk.get_hit(HITId=hit["HITId"])
    q = xmltodict.parse(hit["Question"])
    x = q["HTMLQuestion"]["HTMLContent"]
    q2 = xmltodict.parse(x)

    if worker_results['NumResults'] > 0:
      for assignment in worker_results['Assignments']:
        tmp = answer_to_dict(assignment['Answer'])
        mos_ratings = parse_mos_answers(tmp)
        lt = parse_listening_type(tmp)
        comment = parse_comment(tmp)

    else:
      # print("No results ready yet")
      pass
