import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--run-performance',
        action='store_true',
        default=False,
        help='run timing-sensitive performance tests',
    )


def pytest_collection_modifyitems(config, items):
    enabled = (config.getoption('--run-performance') or
               os.environ.get('TMM_FAST_RUN_PERFORMANCE') == '1')
    if enabled:
        return

    skip = pytest.mark.skip(reason='use --run-performance to run performance tests')
    for item in items:
        if 'performance' in item.keywords:
            item.add_marker(skip)
