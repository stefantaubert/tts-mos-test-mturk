from collections import OrderedDict

from pandas import DataFrame

from mturk_template.convert_to_json import convert_to_json


def test_component():
  gt_df = DataFrame([
      ["url1", "alg1", "file1"],
      ["url2", "alg1", "file2"],
      ["url3", "alg2", "file1"],
      ["url4", "alg2", "file2"],
      ["url5", "alg3", "file3"],
    ],
    columns=[
      "audio_url", "algorithm", "file"
    ]
  )

  res_df = DataFrame([
      [
        "hit0", "assignment0", "worker00", "Approved", "37",
        "url1", "url2",
        "comment",
        False, False, True, False,  # Listening type
        True, False, False, False, False,  # MOS url1
        False, False, False, False, True,  # MOS url2
      ],
      [
        "hit1", "assignment1", "worker01", "Approved", "38",
        "url3", "url4",
        "comment",
        False, False, False, True,  # Listening type
        False, True, False, False, False,  # MOS url1
        False, False, True, False, False,  # MOS url2
      ],
    ],
    columns=[
      "HITId", "AssignmentId", "WorkerId", "AssignmentStatus", "WorkTimeInSeconds",
      "Input.audio_url_0", "Input.audio_url_1",
      "Answer.comments",
      "Answer.listening-type.over-the-ear", "Answer.listening-type.in-ear", "Answer.listening-type.desktop", "Answer.listening-type.laptop",
      "Answer.0-mos-rating.1", "Answer.0-mos-rating.2", "Answer.0-mos-rating.3", "Answer.0-mos-rating.4", "Answer.0-mos-rating.5",
      "Answer.1-mos-rating.1", "Answer.1-mos-rating.2", "Answer.1-mos-rating.3", "Answer.1-mos-rating.4", "Answer.1-mos-rating.5",
    ]
  )

  result = convert_to_json(res_df, gt_df)

  assert result == OrderedDict([
    ('algorithms', ['alg1', 'alg2', 'alg3']),
    ('files', ['file1', 'file2', 'file3']),
    ('workers', OrderedDict([
      ('worker00', OrderedDict([
        ('assignment0', OrderedDict([
          ('device', 'desktop'),
          ('state', 'Approved'),
          ('worktime', 37),
          ('ratings', [
            OrderedDict([
              ('algorithm', 'alg1'),
              ('file', 'file1'),
              ('rating', 1)]),
            OrderedDict([
              ('algorithm', 'alg1'),
              ('file', 'file2'),
              ('rating', 5)
            ])
          ])
        ]))
      ])),
      ('worker01', OrderedDict([
        ('assignment1', OrderedDict([
          ('device', 'laptop'),
          ('state', 'Approved'),
          ('worktime', 38),
          ('ratings', [
            OrderedDict([
              ('algorithm', 'alg2'),
              ('file', 'file1'),
              ('rating', 2)
            ]),
            OrderedDict([
              ('algorithm', 'alg2'),
              ('file', 'file2'),
              ('rating', 3)
            ])
          ])
        ])
        )])
       )]
    ))])