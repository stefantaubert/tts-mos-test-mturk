from tts_mos_test_mturk.logging import (attach_boto_to_detail_logger,
                                        attach_urllib3_to_detail_logger, get_detail_logger,
                                        get_logger)
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path, parse_positive_integer
from tts_mos_test_mturk_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                          init_and_return_loggers,
                                                          try_init_file_buffer_logger)
from tts_mos_test_mturk_cli.parsers.api.api_approve_parser import get_api_approve_parser
from tts_mos_test_mturk_cli.parsers.api.api_bonus_parser import get_api_bonus_parser
from tts_mos_test_mturk_cli.parsers.api.api_reject_parser import get_api_reject_parser
from tts_mos_test_mturk_cli.parsers.api.approve_parser import get_approve_parser
from tts_mos_test_mturk_cli.parsers.api.bonus_parser import get_bonus_parser
from tts_mos_test_mturk_cli.parsers.api.reject_parser import get_reject_parser
from tts_mos_test_mturk_cli.parsers.statistics.export_ground_truth_parser import get_export_gt_parser
from tts_mos_test_mturk_cli.parsers.init_parser import get_init_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_assignments_by_listening_device_parser import \
  get_mask_assignments_by_listening_device_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_assignments_by_work_time_parser import \
  get_mask_assignments_by_work_time_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_outlying_scores_parser import \
  get_mask_outlying_scores_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_scores_by_masked_count_parser import \
  get_mask_scores_by_masked_count_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_workers_by_assignment_count_parser import \
  get_mask_workers_by_assignment_count_parser
from tts_mos_test_mturk_cli.parsers.masking.mask_workers_by_correlation_parser import (
  get_mask_workers_by_correlation_parser, get_mask_workers_by_correlation_percent_parser)
from tts_mos_test_mturk_cli.parsers.statistics.calculation_parser import get_calculation_parser
from tts_mos_test_mturk_cli.parsers.statistics.export_algorithm_sentence_stats import \
  get_export_as_stats_parser
from tts_mos_test_mturk_cli.parsers.statistics.export_algorithm_worker_stats import \
  get_export_aw_stats_parser
from tts_mos_test_mturk_cli.parsers.statistics.export_worker_assignment_stats_parser import \
  get_export_wa_stats_parser
from tts_mos_test_mturk_cli.parsers.statistics.stats_parser import get_stats_parser
