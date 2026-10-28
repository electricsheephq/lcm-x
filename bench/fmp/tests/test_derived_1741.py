"""Stage 0 accounting regressions; fabricated inputs and fake transports only."""
import json

import pytest

from bench.fmp.tests.test_derived import (
    Fake, bundle, cache, extract, item, run, score, window, wire,
)


def test_empty_repair_keeps_rejections_above_semantic_cap(tmp_path):
    class EmptyRepair(Fake):
        def call(self, *args, **kw):
            response = super().call(*args, **kw)
            assert kw['max_tokens'] == 16384, 'invalid extraction must stop before reader/judge calls'
            output = json.loads(response['content'])
            for raw in output:
                raw['items'] = [item(), item('bad') | dict(value_text='unquoted')] if self.calls == 1 else []
            response['content'] = json.dumps(output)
            return response

    run.freeze(tmp_path, bundle())
    backend = EmptyRepair()
    report = run.execute(tmp_path, backend, stage='extract')
    extracted = cache.snapshot(tmp_path/'extraction.json')
    assert len(extracted['items']) == 1
    ledger = extracted['ledger'][0]
    assert ledger['state'] == 'partial'
    assert ledger['rejected_count'] == 1
    assert ledger['rejected'] == [dict(item=1, reason='source/type admission')]
    evaluation = cache.snapshot(tmp_path/'evaluation-inputs.json')
    assert evaluation['parameters']['semantic_valid'] is False
    assert report['verdict'] == 'PARK-INCOMPLETE'
    assert report['rejected_items'] == 1 and report['extracted_items'] == 2
    assert report['rejection_rate'] == .5
    assert backend.calls == 2


def test_empty_repair_keeps_rejected_relation():
    w = window()
    original = wire(w, [item()], [dict(from_local_id='a', to_local_id='missing')])
    admitted = extract.validate(original, w)
    targets = extract.repair_targets([w], [original], [admitted])[0]
    repaired = extract.replacements(wire(w), original, targets)
    assert repaired['items'] == original['items']
    assert repaired['relations'] == original['relations']
    assert extract.validate(repaired, w) == admitted
    assert admitted['state'] == 'partial' and admitted['rejected'] == [
        dict(relation=0, reason='relation admission')]


@pytest.fixture
def expiring_repair(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(cache.time, 'time', lambda: clock[0])
    b = bundle() | dict(deadline=1010.)
    run.freeze(tmp_path, b)

    class Expiring(Fake):
        def call(self, *args, **kw):
            response = super().call(*args, **kw)
            assert self.calls == 1 and kw['max_tokens'] == 16384
            output = json.loads(response['content'])
            output[0]['items'].append(item('bad') | dict(value_text='unquoted'))
            response['content'] = json.dumps(output)
            clock[0] = 1020.
            return response

    return Expiring()


def test_repair_deadline_retains_items_and_never_completes_extract(tmp_path, expiring_repair):
    """Lock-in: already green on the base; the repair call reaches the outer handler."""
    report = run.execute(tmp_path, expiring_repair, stage='extract')
    extracted = cache.snapshot(tmp_path/'extraction.json')
    assert extracted['terminal_reason'] == 'deadline expired'
    expected = extract.validate(wire(window(), [item()]), window())['items']
    assert extracted['items'] == expected
    assert extracted['ledger'][0]['state'] == 'failed'
    assert report.get('status') != 'COMPLETE'
    assert report['verdict'] == 'PARK-INCOMPLETE'
    assert report['reasons'] == ['deadline expired']
    assert expiring_repair.calls == 1, 'expired repair must not send another request'


def test_extraction_deadline_saved_incomplete_and_rescores(tmp_path, expiring_repair):
    report = run.execute(tmp_path, expiring_repair)
    evaluation = cache.snapshot(tmp_path/'evaluation-inputs.json')
    assert evaluation['parameters']['incomplete_reason'] == 'deadline expired'
    assert evaluation['parameters']['complete'] is False
    rows = cache.snapshot(tmp_path/'rows.json')
    assert score.evaluate(evaluation['entries'], rows, **evaluation['parameters']) == report
    assert cache.snapshot(tmp_path/'report.json') == report
    assert run.execute(tmp_path, expiring_repair) == report
