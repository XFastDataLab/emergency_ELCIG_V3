"""Verify the single Self-Refine attempt independently of the ELCIG limit."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config
from elcig import ELCIGLoop
from models import ErrorRecord, Script, ScriptStep


class SelfRefineTests(unittest.TestCase):
    def run_loop(self, response, limit=5):
        cfg = Config.from_dict({
            'rag': {'use_rag': False}, 'validator': {'use_validator': False},
            'correction': {'mode': 'self_refine'},
            'experiment': {'max_iterations': limit},
        })
        validator = Mock()
        validator.validate.return_value = [ErrorRecord('resource', 1, '位置', '仍有错误', '规则')]
        validator.constraint_count.return_value = 10
        client = Mock()
        client.chat.return_value = response
        loop = ELCIGLoop(cfg, None, None, validator, None, client, None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            final, history = loop.run('S1', initial_script=Script('S1', [ScriptStep(1, action='启动响应')]))
        return client, validator, final, history, output.getvalue()

    def test_only_one_attempt_even_when_errors_remain(self):
        response = json.dumps({'needs_fix': True, 'steps': [
            {'step_id': 1, 'action': '人员疏散', 'role': '消防救援组'}]})
        for limit in (1, 5, 10):
            with self.subTest(limit=limit):
                client, validator, final, history, output = self.run_loop(response, limit)
                client.chat.assert_called_once()
                self.assertEqual(validator.validate.call_count, 2)
                self.assertEqual(len(history), 2)
                self.assertEqual(history[-1]['error_count'], 1)
                self.assertEqual(final.steps[0].action, '人员疏散')
                self.assertIn('Self-Refine 迭代 1/1', output)

    def test_no_fix_or_invalid_response_does_not_retry(self):
        for response in ('{}', '{"needs_fix": false}', '{"needs_fix": true}'):
            with self.subTest(response=response):
                client, validator, _, history, _ = self.run_loop(response)
                client.chat.assert_called_once()
                validator.validate.assert_called_once()
                self.assertEqual(len(history), 1)

    def test_zero_limit_disables_attempt(self):
        client, _, _, history, _ = self.run_loop('{}', 0)
        client.chat.assert_not_called()
        self.assertEqual(len(history), 1)

    def test_elcig_keeps_multiple_correction_rounds(self):
        cfg = Config.from_dict({
            'rag': {'use_rag': False}, 'validator': {'use_validator': True},
            'correction': {'mode': 'rule_feedback'},
            'experiment': {'max_iterations': 3},
        })
        validator = Mock()
        validator.validate.return_value = [ErrorRecord('resource', 1, '位置', '仍有错误', '规则')]
        validator.constraint_count.return_value = 10
        corrector = Mock()
        corrector.correct.side_effect = lambda script, errors, scenario: script
        loop = ELCIGLoop(cfg, None, None, validator, corrector, None, None)
        with contextlib.redirect_stdout(io.StringIO()):
            _, history = loop.run('S1', initial_script=Script('S1'))
        self.assertEqual(corrector.correct.call_count, 3)
        self.assertEqual(len(history), 4)
