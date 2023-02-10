from tts_mos_test_mturk_cli.cli import create_debug_file, parse_args

create_debug_file()

args = ['reject', '/tmp/data.pkl', 'bad work', '/tmp/reject.csv']
# args.extend(['-j', "1"])
# args.append("--debug")

parse_args(args)
