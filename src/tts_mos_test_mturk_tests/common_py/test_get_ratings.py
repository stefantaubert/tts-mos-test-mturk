# from collections import OrderedDict
# from datetime import datetime

# import numpy as np
# from ordered_set import OrderedSet

# from tts_mos_test_mturk.common import get_ratings
# from tts_mos_test_mturk.evaluation_data import EvaluationData
# from tts_mos_test_mturk.result import Assignment, Result, Worker


# def test_get_first_rating():
#   r = Result(
#     algorithms=OrderedSet(["alg1", "alg2"]),
#     files=OrderedSet(["file1", "file2"]),
#     workers=OrderedDict([
#       ("worker1", Worker(
#         age_group="18-29",
#         gender="male",
#         assignments=OrderedDict([
#           ("assignment1", Assignment(
#             device="headphone",
#             state="Pending",
#             worktime=30,
#             comments="",
#             hit_id="hit1",
#             time=datetime.now(),
#             ratings=OrderedDict([
#               (("alg1", "file1"), OrderedDict([
#                 ("naturalness", 5),
#                 ("intelligibility", 3),
#               ])),
#               (("alg2", "file2"), OrderedDict([
#                 ("naturalness", 4),
#                 ("intelligibility", 2),
#               ])),
#             ])
#           )),
#           ("assignment2", Assignment(
#             device="headphone",
#             state="Pending",
#             worktime=30,
#             comments="",
#             hit_id="hit2",
#             time=datetime.now(),
#             ratings=OrderedDict([
#               (("alg1", "file2"), OrderedDict([
#                 ("naturalness", 5),
#                 ("intelligibility", 3),
#               ])),
#               (("alg2", "file2"), OrderedDict([
#                 ("naturalness", 3),  # duplicate -> (3 + 4) / 2 = 3.5
#                 ("intelligibility", 2),
#               ])),
#             ])
#           ))
#         ])
#       ))
#     ])
#   )

#   data = EvaluationData(r)
#   result = get_ratings(data, {"naturalness"})

#   assert_result = np.array(
#     # worker1
#     [
#       # alg1
#       [
#         # file1, file2
#         [5.0, 5.0]
#       ],
#       # alg2
#       [
#         # file1, file2
#         [np.nan, 3.5]
#       ]
#     ]
#   )

#   np.testing.assert_array_equal(assert_result, result)
