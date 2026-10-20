import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config
from elcig import ELCIGLoop
from experiments import run_method
from models import ErrorRecord, Script


def errors(count, start=0, kind='resource'):
    return [ErrorRecord(kind, i, f'步骤{i}', f'错误{i}', f'规则{i}')
            for i in range(start, start + count)]


class FeedbackTests(unittest.TestCase):
    def run_loop(self, batches, a=4, max_iter=5):
        cfg = Config.from_dict({
            'rag': {'use_rag': False}, 'validator': {'use_validator': True},
            'correction': {'mode': 'rule_feedback'},
            'experiment': {'max_iterations': max_iter},
        })
        validator = Mock()
        validator.validate.side_effect = batches
        validator.constraint_count.return_value = 100
        corrector = Mock()
        corrector.correct.side_effect = lambda script, feedback, scenario: script
        loop = ELCIGLoop(cfg, None, None, validator, corrector, None, None)
        with patch('elcig.random.randint', return_value=a) as randint, \
             patch('elcig.random.sample', return_value=list(range(a))) as sample, \
             patch('elcig.emit') as emit, contextlib.redirect_stdout(io.StringIO()):
            _, history = loop.run('S1', initial_script=Script('S1'))
        feedback = [c.args[1] for c in corrector.correct.call_args_list]
        events = [c.kwargs for c in emit.call_args_list if c.args[0] == 'correction_start']
        return feedback, history, randint, sample, events

    def test_deferred_plus_detected_then_all_even_above_threshold(self):
        initial = errors(12)
        detected = errors(3, 100)
        later = errors(15, 200)
        feedback, history, randint, sample, events = self.run_loop(
            [initial, detected, later, []])
        self.assertEqual(feedback, [initial[:4], initial[4:] + detected, later])
        randint.assert_called_once_with(1, 9)
        sample.assert_called_once_with(range(12), 4)
        self.assertEqual([h['error_count'] for h in history], [12, 3, 15, 0])
        self.assertEqual([h['pending_error_count'] for h in history], [12, 11, 15, 0])
        self.assertEqual([h['deferred_error_count'] for h in history], [0, 8, 0, 0])
        self.assertEqual([e['feedback_error_count'] for e in events], [4, 11, 15])
        self.assertEqual(events[0]['deferred_errors'], [e.to_dict() for e in initial[4:]])

    def test_deferred_still_sent_when_next_validation_is_clean(self):
        initial = errors(11)
        feedback, history, *_ = self.run_loop([initial, [], []])
        self.assertEqual(feedback, [initial[:4], initial[4:]])
        self.assertEqual([h['pending_error_count'] for h in history], [11, 7, 0])

    def test_merge_preserves_duplicates_to_match_n_minus_a_plus_m(self):
        initial = errors(11)
        feedback, *_ = self.run_loop([initial, [initial[-1]], []])
        self.assertEqual(feedback[1], initial[4:] + [initial[-1]])

    def test_threshold_and_hints_and_later_large_batch_do_not_sample(self):
        initial = errors(10) + errors(4, 30, 'temporal_hint')
        later = errors(12, 100)
        feedback, history, randint, sample, _ = self.run_loop([initial, later, []])
        self.assertEqual(feedback, [initial[:10], later])
        self.assertEqual([h['pending_error_count'] for h in history], [10, 12, 0])
        randint.assert_not_called()
        sample.assert_not_called()

    def test_no_hard_errors_stops_without_feedback(self):
        feedback, _, randint, _, _ = self.run_loop([errors(12, kind='responsibility_hint')])
        self.assertEqual(feedback, [])
        randint.assert_not_called()

    def test_random_count_endpoints_and_iteration_limit(self):
        initial = errors(11)
        for a in (1, 8):
            with self.subTest(a=a):
                feedback, history, randint, _, events = self.run_loop([initial, []], a=a, max_iter=1)
                self.assertEqual(feedback, [initial[:a]])
                self.assertEqual([h['pending_error_count'] for h in history], [11, 11-a])
                self.assertEqual(events[0]['deferred_error_count'], 11-a)
                randint.assert_called_once_with(1, 8)

    def test_result_json_uses_pending_curve_and_keeps_detection_metrics(self):
        import json
        _, history, *_ = self.run_loop([errors(12), errors(3, 100), errors(2, 200), []])
        script = Script('S1')
        loop = Mock(last_knowledge='')
        loop.run.return_value = script, history
        cfg = Config.from_dict({'rag': {'use_rag': True}, 'validator': {'use_validator': True}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / 'shared_RAG_draft'
            source.mkdir()
            draft = dict(script=script, knowledge='', repeat=1, use_rag=True,
                         source_dir=str(source), draft_id='test', draft_hash='test',
                         generation_seconds=0, generation_cost={})
            with patch('experiments._loop', return_value=(cfg, loop)):
                run_method(cfg, 'ELCIG', 'S1', {}, shared_draft=draft, output_dir=path)
            result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['error_curve'], [12, 11, 2, 0])
        self.assertEqual(result['detected_error_curve'], [12, 3, 2, 0])
        self.assertEqual(result['history'][1]['csr'], 0.97)
        self.assertEqual(result['final_errors'], 0)


if __name__ == '__main__':
    unittest.main()
