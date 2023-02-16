
import pandas as pd
from mypy_boto3_mturk import MTurkClient
from tqdm import tqdm

from tts_mos_test_mturk.logging import get_detail_logger


def grant_bonuses_from_df(df: pd.DataFrame, mturk: MTurkClient) -> bool:
  dlogger = get_detail_logger()
  all_successful = True
  for i, row in tqdm(list(df.iterrows()), desc="Sending bonuses", unit=" assignment(s)"):
    dlogger.info(
      f"Sending a {row['BonusAmount']:.2f}$ bonus to worker \"{row['WorkerId']}\" for assignment \"{row['AssignmentId']}\" with reason \"{row['Reason']}\".")
    try:
      mturk.send_bonus(
        WorkerId=row["WorkerId"],
        BonusAmount=str(row["BonusAmount"]),
        Reason=row["Reason"],
        AssignmentId=row["AssignmentId"],
      )
    except Exception as ex:
      dlogger.error(f"Bonus couldn't be send!")
      dlogger.debug(ex, exc_info=True)
      all_successful = False
      continue
  return all_successful


def approve_from_df(df: pd.DataFrame, mturk: MTurkClient) -> bool:
  dlogger = get_detail_logger()
  all_successful = True
  for i, row in tqdm(list(df.iterrows()), desc="Approving", unit=" assignment(s)"):
    dlogger.info(
      f"Approving assignment \"{row['AssignmentId']}\" of worker \"{row['WorkerId']}\" with reason \"{row['Approve']}\".")
    try:
      mturk.approve_assignment(
        AssignmentId=row["AssignmentId"],
        RequesterFeedback=row["Approve"],
      )
    except Exception as ex:
      dlogger.error(f"Assignment couldn't be approved!")
      dlogger.debug(ex, exc_info=True)
      all_successful = False
      continue
  return all_successful


def reject_from_df(df: pd.DataFrame, mturk: MTurkClient) -> bool:
  dlogger = get_detail_logger()
  all_successful = True
  for i, row in tqdm(list(df.iterrows()), desc="Rejecting", unit=" assignment(s)"):
    dlogger.info(
      f"Rejecting assignment \"{row['AssignmentId']}\" of worker \"{row['WorkerId']}\" with reason \"{row['Reject']}\".")
    try:
      mturk.reject_assignment(
        AssignmentId=row["AssignmentId"],
        RequesterFeedback=row["Reject"],
      )
    except Exception as ex:
      dlogger.error(f"Assignment couldn't be rejected!")
      dlogger.debug(ex, exc_info=True)
      all_successful = False
      continue
  return all_successful
