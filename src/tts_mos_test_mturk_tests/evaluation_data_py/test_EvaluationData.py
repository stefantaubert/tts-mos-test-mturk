# import numpy as np
# from ordered_set import OrderedSet
# from pandas import DataFrame

# from tts_mos_test_mturk.common import get_ratings
# from tts_mos_test_mturk.evaluation_data import EvaluationData
# from tts_mos_test_mturk.result import parse_result_from_json

# _ = np.nan


# def test_component():
#   gt_df = DataFrame([
#       ["url1", "alg1", "file1"],
#       ["url2", "alg1", "file2"],
#       ["url3", "alg2", "file1"],
#       ["url4", "alg2", "file2"],
#       ["url5", "alg3", "file3"],
#     ],
#     columns=[
#       "audio_url", "algorithm", "file"
#     ]
#   )

#   res_df = DataFrame([
#       [
#         "hit0", "assignment0", "worker00", "Approved", "37",
#         "url1", "url2",
#         "comment",
#         False, False, True, False,  # Listening type
#         True, False, False, False, False,  # MOS url1
#         False, False, False, False, True,  # MOS url2
#       ],
#       [
#         "hit1", "assignment1", "worker01", "Approved", "38",
#         "url3", "url4",
#         "comment",
#         False, False, False, True,  # Listening type
#         False, True, False, False, False,  # MOS url1
#         False, False, True, False, False,  # MOS url2
#       ],
#     ],
#     columns=[
#       "HITId", "AssignmentId", "WorkerId", "AssignmentStatus", "WorkTimeInSeconds",
#       "Input.audio_url_0", "Input.audio_url_1",
#       "Answer.comments",
#       "Answer.listening-type.over-the-ear", "Answer.listening-type.in-ear", "Answer.listening-type.desktop", "Answer.listening-type.laptop",
#       "Answer.0-mos-rating.1", "Answer.0-mos-rating.2", "Answer.0-mos-rating.3", "Answer.0-mos-rating.4", "Answer.0-mos-rating.5",
#       "Answer.1-mos-rating.1", "Answer.1-mos-rating.2", "Answer.1-mos-rating.3", "Answer.1-mos-rating.4", "Answer.1-mos-rating.5",
#     ]
#   )

#   result = convert_to_json(res_df, gt_df)
#   r = parse_result_from_json(result)
#   data = EvaluationData(r)

#   assert data.n_workers == 2
#   assert data.workers == OrderedSet(("worker00", "worker01"))

#   assert data.n_algorithms == 3
#   assert data.algorithms == OrderedSet(("alg1", "alg2", "alg3"))

#   assert data.n_files == 3
#   assert data.files == OrderedSet(("file1", "file2", "file3"))

#   assert data.n_assignments == 2
#   assert data.assignments == OrderedSet(("assignment0", "assignment1"))

#   np.testing.assert_equal(get_ratings(data), [
#     [
#       [1, 5, _],
#       [_, _, _],
#     ],
#     [
#       [_, _, _],
#       [2, 3, _],
#     ],
#     [
#       [_, _, _],
#       [_, _, _],
#     ],
#   ])
