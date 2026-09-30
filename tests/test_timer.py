"""
Test the context timer without waiting for real time to pass.
"""
import datetime
from unittest.mock import patch

import pytest
import torchtt


@pytest.mark.parametrize("name", ['', 'test'])
@pytest.mark.parametrize("seconds, output", [(0, '0 ns'), (1e-6, '1 us'), (1e-3, '1 ms'), (1, '1 s')])
def test_timer(name, seconds, output, capsys):
    start = datetime.datetime(2026, 1, 1)
    with patch('torchtt._custom_timer.datetime') as clock:
        clock.datetime.now.side_effect = [start, start + datetime.timedelta(seconds=seconds)]
        with torchtt.Timer(name) as timer:
            assert timer.start == start

    label = ' "test"' if name else ''
    assert timer.interval == seconds
    assert capsys.readouterr().out == f'Timer{label} took {output}\n'


def test_timer_exception(capsys):
    with pytest.raises(ValueError, match='test error'):
        with torchtt.Timer():
            raise ValueError('test error')
    assert 'Timer took' in capsys.readouterr().out
