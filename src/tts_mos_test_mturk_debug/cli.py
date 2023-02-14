from tts_mos_test_mturk_cli.cli import create_debug_file, parse_args

create_debug_file()

args =  ['stats-worker-assignments', '/tmp/crowdMOS.pkl', '/tmp/worker-assignments.csv', '-m', 'fast-workers', 'bad-workers', 'fast-workers & bad-workers -> no-headphones', 'fast-workers & bad-workers -> no-headphones -> outliers-1', 'fast-workers & bad-workers -> no-headphones -> outliers-2']
# args.extend(['-j', "1"])
# args.append("--debug")

parse_args(args)
