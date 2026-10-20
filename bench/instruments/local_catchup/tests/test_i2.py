import pytest

from bench.instruments.local_catchup.turns import paired_bar
from bench.instruments.local_catchup.recall import overlapping, score
from bench.instruments.local_catchup.floor import command, rate, verify_limits


@pytest.mark.parametrize('off,on,limit,state', [(100,150,50,'PASS'), (1000,1100,100,'PASS'), (100,151,50,'FAIL'), (500,550,50,'PASS')])
def test_paired_bar(off, on, limit, state):
    result = paired_bar([(off,on)]*3)
    assert result['limit_ms'] == limit
    assert result['state'] == state
    assert result['upper95_ms'] == on-off


def test_paired_bootstrap_preserves_matching_and_seed():
    pairs = [(10,20),(100,110),(1000,1010)]
    assert paired_bar(pairs)['upper95_ms'] == 10
    assert paired_bar(pairs) == paired_bar(pairs)
    assert paired_bar([(1,1),(1,301)])['state'] == 'FAIL'


def row(start=1, end=2, seconds=.1, health='ok'):
    return dict(start=start, end=end, seconds=seconds, health=health, fts_hits=True, honest_coverage=True)


def test_overlap_excludes_outside_and_touching():
    inside, outside, touching = row(), row(3,4), row(2,3)
    assert overlapping([inside,outside,touching], [dict(start=1.5,end=2)]) == [inside]


@pytest.mark.parametrize('recalls,pubs', [(199,10),(200,9),(0,10)])
def test_invalid_volume(recalls, pubs):
    assert score([row()]*recalls, [dict(start=1,end=2)]*pubs)['state'] == 'INVALID'


@pytest.mark.parametrize('seconds,health,expected', [(2,'ok','PASS'), (2.001,'ok','FAIL'), (.1,'timeout','FAIL'), (.1,'error','FAIL')])
def test_recall_bars_and_timeout(seconds, health, expected):
    result = score([row(seconds=seconds,health=health)]*200, [dict(start=1,end=2)]*10)
    assert result['state'] == expected
    assert result['statistics']['p90']['upper95_s'] == seconds


def test_p90_and_disclosure_failures():
    rows = [row(seconds=.1)]*160+[row(seconds=8.01)]*40
    assert score(rows, [dict(start=1,end=2)]*10)['state'] == 'FAIL'
    bad = row()
    bad['honest_coverage'] = False
    assert score([bad]*200, [dict(start=1,end=2)]*10)['state'] == 'FAIL'


def test_floor_command_and_effective_limits():
    cmd = command(['--cell','recall'])
    assert cmd[:3] == ['taskset','-c','0,1']
    assert cmd[-1] == '--limited'
    assert verify_limits('4294967296\n','0-1\n')['verified']
    for memory, cpus in [('max','0-1'),('4294967296','0-3'),('2147483648','0,1')]:
        with pytest.raises(RuntimeError):
            verify_limits(memory,cpus)


def test_floor_rate_needs_sustained_incomplete_two_hours():
    assert rate(dict(state='INCOMPLETE',rows_s=10),7200,250000)['state'] == 'PASS'
    assert rate(dict(state='COMPLETE',rows_s=10),7200,250000)['state'] == 'INVALID'
    assert rate(dict(state='INCOMPLETE',rows_s=.1),7200,250000)['state'] == 'FAIL'
