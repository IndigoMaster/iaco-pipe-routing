"""
Entry point.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pipe_router.parser import parse_yaml
from pipe_router.scenario import PipeRoutingScenario
from viz.visualization import Visualizer

LOG_FILE_NAME = 'log.txt'


def execute() -> None:
    """
    Method of execution.
    """
    params = parse_yaml(config_file_path)
    params.random_seed = seed
    params.config_file_path = config_file_path
    params.quiet = quiet
    params.output_dir_path = output_dir_path

    scenario = PipeRoutingScenario(params)
    results = scenario.solve()

    viz = Visualizer.from_scenario(scenario, results)
    viz.render(show=show_plot)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('yaml_path',
                        type=str,
                        help='Abs path to input file')
arg_parser.add_argument('-o', '--outdir',
                        type=str,
                        help='Abs path to output dir')
arg_parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Quiet (no output)')
arg_parser.add_argument('-s', '--seed',
                        type=int,
                        help='Random seed')
arg_parser.add_argument('--show',
                        action='store_true',
                        help='Show interactive plot (blocking call)')
arg_parser.add_argument('--loglevel',
                        type=str,
                        default='warn',
                        help='Log level: debug, info, warn')
arg_parser.add_argument('--streamlog',
                        action='store_true',
                        help='Show interactive plot (blocking call)')

arg_parser.prog = 'pipe_router'
args = arg_parser.parse_args()

# arg: quiet mode
quiet = (args.quiet is True)

# arg: debug
debug = (args.debug is True)

# arg: log level
log_formatter = logging.Formatter('%(levelname)s | %(message)s')
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().handlers.clear()
if not quiet:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    log_level = logging.WARNING if not args.streamlog else args.loglevel
    stream_handler.setLevel(log_level)
    logging.getLogger().addHandler(stream_handler)

try:
    # arg: random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(np.random.rand() * (2 ** 31))
    logging.info(f'Random seed: {seed}')
    np.random.seed(seed)

    # arg: path to yaml file
    config_file_path = Path(args.yaml_path).resolve()
    logging.info(f'Path to YAML config file: {config_file_path}')
    if not config_file_path.exists():
        raise ValueError(f'YAML config file does not exist: {config_file_path}')

    # arg: output dir
    output_dir_path = None
    if args.outdir is not None:
        output_dir_path = Path(args.outdir)
        if not output_dir_path.exists():
            raise ValueError(f'Output directory does not exist: {output_dir_path}')
        if not output_dir_path.is_dir():
            raise ValueError(f'Output directory is a file, not a directory: {output_dir_path}')
    if output_dir_path is None:
        logging.info('Output dir path is not specified. Logging to stdout.')
    else:
        log_path = Path(output_dir_path / LOG_FILE_NAME).resolve()
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(args.loglevel)
        logging.getLogger().addHandler(file_handler)
        logging.info('Output dir path is {output_dir_path}')

    # arg: show
    show_plot = args.show
    if show_plot:
        logging.info('Running with --show. An interactive plot will be displayed after solving.')
    else:
        logging.info('Running without --show. No interactive plot will be displayed.')

    # =============================================================================

    execute()

except Exception as e:
    logging.error(f'{e}\n\nAborted\n')
    logging.exception(e)

finally:
    logging.shutdown()
