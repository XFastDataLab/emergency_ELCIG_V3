"""Offline regressions; all generated artifacts remain under output/ALL-tests-* ."""
import contextlib
import csv
from collections import Counter
from datetime import datetime, timezone, timedelta
import copy
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import audit
import experiments
from audit import AuditedClient
from config import Config
from evaluation import VALIDATOR_INJECTION_PLAN, build_boundary_cases, build_resource_boundary_case, build_validator_cases, validator_dataset_effectiveness
from generator import LLMGenerator
from models import ResourceDemand, Script, ScriptStep
from scenarios import get_scenario
from validator.responsibility_validator import ResponsibilityValidator
from validator.composite import RuleValidator
from rules.registry import get_rules
from rules.matching import three_character_match
from action_grounding import semantic_match
from parser import steps_from_json
from validator_report import HEADERS, build_comparison_report, save_comparison_report

TEST_OUTPUT = audit.new_directory('ALL','tests')


class FakeClient:
    def __init__(self):
        self.generation_calls = 0
        self.calls = []

    def chat(self,system,user,json_mode=False):
        self.calls.append((system,user))
        if 'needs_fix' in system:
            return json.dumps({'needs_fix': True, 'steps': get_scenario('S1')['correct_base']},ensure_ascii=False)
        if '脚本修正专家' in system:
            return json.dumps({'steps':get_scenario('S1')['correct_base']},ensure_ascii=False)
        self.generation_calls += 1
        steps = copy.deepcopy(get_scenario('S1')['naive_draft'])
        steps[0]['description'] = f'独立生成编号{self.generation_calls}'
        return json.dumps({'steps':steps},ensure_ascii=False)


class FakeRetriever:
    calls = 0
    def is_available(self): return True
    def retrieve_chunks(self,query,top_k=5):
        self.calls += 1
        return [{'title':'测试参考','content':'TEST_RETRIEVED_CONTEXT','query':query}]


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.from_file(ROOT/'config.yaml')
        self.cfg.experiment['scenarios'] = ['S1']
        self.cfg.experiment['max_iterations'] = 1
        self.cfg.experiment['save_intermediate'] = False

    def test_two_shared_drafts_each_repeat_and_full_audit(self):
        fake = FakeClient()
        client = AuditedClient(fake)
        retriever = FakeRetriever()
        comps = {'client':client,'retriever':retriever,'generator':LLMGenerator(client)}
        with patch.object(audit,'OUTPUT',TEST_OUTPUT), patch.object(experiments,'build_components',return_value=comps), contextlib.redirect_stdout(io.StringIO()):
            rows = experiments.run_ablation(self.cfg,repeats=2)
        self.assertEqual(fake.generation_calls,4)
        self.assertEqual(len(rows),8)
        self.assertFalse(any(TEST_OUTPUT.glob('*-shared-RAG-*')))
        self.assertFalse(any(TEST_OUTPUT.glob('*-shared-noRAG-*')))
        self.assertEqual(retriever.calls,10)  # five queries for each of two RAG drafts
        self.assertEqual(len({r['draft_id'] for r in rows}),4)
        for repeat in (1,2):
            group = {r['method_key']:r for r in rows if r['repeat']==repeat}
            for a,b in [('LLM','SelfRefine'),('RAG','ELCIG')]:
                self.assertEqual(group[a]['draft_hash'],group[b]['draft_hash'])
                self.assertEqual(group[a]['history'][0]['script'],group[b]['history'][0]['script'])
            self.assertFalse(group['SelfRefine']['use_rag'])
            self.assertEqual(group['SelfRefine']['knowledge'],'')
            self.assertEqual(group['LLM']['iterations'],0)
            self.assertEqual(group['RAG']['iterations'],0)
        for row in rows:
            folder = Path(row['output_dir'])
            for name in ['initial.json','final.json','trace.jsonl','history.json','result.json']:
                self.assertTrue((folder/name).exists())
            source = Path(row['shared_source_dir'])
            local = Path(row['shared_draft_copy'])
            for name in ('initial.json', 'metadata.json', 'trace.jsonl', 'llm_messages.json', 'retrieval.json'):
                self.assertEqual((source/name).read_bytes(), (local/name).read_bytes())
            reference = json.loads((folder/'shared_draft_reference.json').read_text(encoding='utf-8'))
            self.assertEqual(folder/reference['local_copy'], local)
            self.assertEqual(reference['draft_hash'], row['draft_hash'])
            events = [json.loads(x) for x in (source/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(x['event']=='llm_request' and x['user'] for x in events))
            self.assertTrue(any(x['event']=='initial_draft' for x in events))
            if row['use_rag']:
                self.assertTrue(any(x['event']=='retrieval_query' and x['chunks'] for x in events))
        reviews = [u for s,u in fake.calls if 'needs_fix' in s]
        self.assertTrue(reviews)
        self.assertTrue(all('TEST_RETRIEVED_CONTEXT' not in u and '验证器反馈' not in u for u in reviews))

    def test_single_method_owns_shared_draft_and_prints_iteration_details(self):
        fake = FakeClient()
        client = AuditedClient(fake)
        comps = {'client': client, 'retriever': FakeRetriever(), 'generator': LLMGenerator(client)}
        folder = TEST_OUTPUT / 'single-method'
        output = io.StringIO()
        with patch.object(audit, 'OUTPUT', folder), contextlib.redirect_stdout(output):
            row = experiments.run_method(self.cfg, 'ELCIG', 'S1', comps)
        owner = Path(row['output_dir'])
        source = Path(row['shared_source_dir'])
        self.assertEqual(source, owner/'shared_RAG_draft')
        self.assertEqual(source, Path(row['shared_draft_copy']))
        self.assertEqual(list(folder.iterdir()), [owner])
        self.assertEqual(fake.generation_calls, 1)
        self.assertTrue((source/'retrieval.json').exists())
        self.assertTrue((source/'llm_messages.json').exists())
        self.assertIn('ELCIG 迭代 1/', output.getvalue())
        self.assertIn('ELCIG 修正｜场景 S1｜AI 输入提示词', output.getvalue())
        self.assertIn('ELCIG 修正｜场景 S1｜AI 输出', output.getvalue())
        self.assertIn('第1轮验证', output.getvalue())
        self.assertTrue(all('hint_count' in step for step in row['history']))

    def test_minute_directory_collisions_preserve_existing_data(self):
        now = datetime(2026, 10, 13, 11, 30, tzinfo=timezone(timedelta(hours=8)))
        with patch.object(audit, 'OUTPUT', TEST_OUTPUT), patch.object(audit, 'datetime') as clock:
            clock.now.return_value = now
            first = audit.new_directory('S1', 'ELCIG')
            (first/'result.json').write_text('keep', encoding='utf-8')
            second = audit.new_directory('S1', 'ELCIG')
            third = audit.new_directory('S1', 'ELCIG')
        self.assertEqual(first.name, 'S1-ELCIG-1013-1130')
        self.assertEqual(second.name, 'S1-ELCIG-1013-1130-02')
        self.assertEqual(third.name, 'S1-ELCIG-1013-1130-03')
        self.assertEqual((first/'result.json').read_text(encoding='utf-8'), 'keep')

    def test_dataset_size_profiles_unique_and_strict_modes(self):
        cases = [case for sid in ('S1','S2','S3') for case in build_validator_cases(sid)]
        self.assertEqual(len(cases),36)
        self.assertEqual(sum(bool(c['expected_errors']) for c in cases),30)
        self.assertEqual(len({json.dumps(c['script'],sort_keys=True) for c in cases}),36)
        self.assertEqual(sum(len(c['expected_errors']) for c in cases),228)
        for sid in ('S1','S2','S3'):
            for strict in (False,True):
                row = validator_dataset_effectiveness(sid,strict=strict,temporal_strict=strict)
                self.assertEqual(row['missed_errors'],0)
                self.assertEqual(row['detected_errors'],row['injected_errors'])
                self.assertTrue(all(row['by_type'][k]['detected_injected_errors']>0 for k in row['by_type']))

    def test_disabled_detector_produces_misses_not_perfect_score(self):
        with patch('evaluation.RuleValidator.validate',return_value=[]):
            row = validator_dataset_effectiveness('S1')
        self.assertEqual(row['ledr'],0)
        self.assertGreater(row['missed_errors'],0)
        self.assertEqual(row['detected_injected_errors'],0)
        self.assertEqual(row['missed_errors'],row['injected_errors'])

    def test_resource_boundary_case_and_controls(self):
        case = build_resource_boundary_case()
        script = Script('S1', [ScriptStep.from_dict(x) for x in case['script']['steps']])
        target = next(s for s in script.steps if len(s.resources) == 2)
        capacity = get_rules('S1').resource['消防车']
        self.assertGreater(sum(r.quantity for r in target.resources), capacity)
        self.assertTrue(all(r.quantity <= capacity for r in target.resources))
        for strict in (False, True):
            detector = RuleValidator(get_rules('S1'), strict=strict, temporal_strict=strict)
            # This regression documents a present limitation, not a target to fix here.
            self.assertEqual(detector.validate(script), [])
            timed = copy.deepcopy(script)
            timed.steps[target.step_id - 1].time = '08:03'
            self.assertTrue(any(e.error_type == 'resource' for e in detector.validate(timed)))
            merged = copy.deepcopy(script)
            merged.steps[target.step_id - 1].resources = [ResourceDemand('消防车', sum(r.quantity for r in target.resources))]
            self.assertTrue(any(e.error_type == 'resource' for e in detector.validate(merged)))
        baseline = validator_dataset_effectiveness('S1')
        stress = validator_dataset_effectiveness('S1', extra_cases=[case])
        self.assertEqual(stress['samples'], 13)
        self.assertEqual(stress['detected_injected_errors'], baseline['detected_injected_errors'])
        self.assertEqual(stress['missed_errors'], 1)
        self.assertEqual(stress['injected_errors'], baseline['injected_errors'] + 1)
        self.assertEqual(stress['cases'][-1]['detected_errors'], [])
        self.assertEqual(stress['cases'][-1]['missed_error_details'],case['expected_errors'])

    def test_additional_detection_does_not_inflate_ledr(self):
        from models import ErrorRecord
        error = ErrorRecord('resource',999,'test','test','不存在的规则')
        with patch('evaluation.RuleValidator.validate',return_value=[error]):
            row = validator_dataset_effectiveness('S1',8)
        self.assertEqual(row['detected_errors'],8)
        self.assertEqual(row['detected_injected_errors'],0)
        self.assertEqual(row['ledr'],0)
        self.assertEqual(row['missed_errors'],row['injected_errors'])
        self.assertTrue(all(c['additional_error_details'] == [error.to_dict()] for c in row['cases']))

    def test_plain_error_counts_and_sample_limit(self):
        self.cfg.experiment['scenarios'] = ['S1','S2','S3']
        with patch.object(audit,'OUTPUT',TEST_OUTPUT):
            rows = experiments.run_validator_effectiveness(self.cfg,include_boundary_case=True)
        self.assertEqual(sum(r['samples'] for r in rows),39)
        self.assertEqual(sum(r['injected_errors'] for r in rows),231)
        self.assertEqual(sum(r['detected_injected_errors'] for r in rows),228)
        self.assertEqual(sum(r['missed_errors'] for r in rows),3)
        removed = {'tp','fp','fn','precision','recall','f1','ledc','clean_false_positive_rate','exact_match_rate'}
        def verify(value):
            if isinstance(value,dict):
                self.assertTrue(removed.isdisjoint(value))
                for child in value.values(): verify(child)
            elif isinstance(value,list):
                for child in value: verify(child)
        verify(rows)
        for row in rows:
            folder = Path(row['output_dir'])
            for name in ('dataset.json','result.json'):
                verify(json.loads((folder/name).read_text(encoding='utf-8')))
            for line in (folder/'trace.jsonl').read_text(encoding='utf-8').splitlines():
                verify(json.loads(line))
        with self.assertRaisesRegex(ValueError,'最多100'):
            experiments.run_validator_effectiveness(self.cfg,samples=100,include_boundary_case=True)
        with self.assertRaisesRegex(ValueError,'最多100'):
            experiments.run_validator_effectiveness(self.cfg,samples=101)

    def test_three_boundary_cases_and_causal_controls(self):
        for sid in ('S1','S2','S3'):
            cases = build_boundary_cases(sid)
            self.assertEqual(len(cases),1)
            case = cases[0]
            self.assertEqual(len(case['expected_errors']),1)
            row = validator_dataset_effectiveness(sid,extra_cases=cases,strict=True,temporal_strict=False)
            self.assertEqual(row['cases'][-1]['missed_errors'],1)
            self.assertIn('miss_reason',row['cases'][-1])
        temporal = build_boundary_cases('S2')[0]
        strict_row = validator_dataset_effectiveness('S2',extra_cases=[temporal],strict=True,temporal_strict=True)
        self.assertEqual(strict_row['cases'][-1]['missed_errors'],0)
        overlap = build_boundary_cases('S3')[0]
        script = Script('S3',[ScriptStep.from_dict(x) for x in overlap['script']['steps']])
        detector = RuleValidator(get_rules('S3'),strict=True,temporal_strict=True)
        self.assertEqual(detector.validate(script),[])
        script.steps[-1].time = script.steps[-2].time
        self.assertTrue(any(e.error_type=='resource' for e in detector.validate(script)))

    def test_comparison_table_groups_and_totals(self):
        rows = [validator_dataset_effectiveness(sid,extra_cases=build_boundary_cases(sid)) for sid in ('S1','S2','S3')]
        report = build_comparison_report(rows)
        self.assertEqual(len(report['rows']),9)
        self.assertEqual(len(report['boundary_explanations']),3)
        self.assertEqual(report['script_samples'],39)
        self.assertEqual(report['total']['error_instances'],231)
        self.assertEqual(report['total']['validator_detected'],228)
        self.assertEqual(report['total']['validator_rate'],round(228/231,4))
        self.assertEqual(len(HEADERS), 5)
        folder = TEST_OUTPUT / 'validator-report'
        folder.mkdir()
        save_comparison_report(folder, rows)
        with (folder/'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
            table = list(csv.reader(stream))
        self.assertEqual(table[0], HEADERS)
        self.assertTrue(all(len(row) == 5 for row in table))
        exported = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(exported, report)
        for item in report['rows']:
            row = next(r for r in rows if r['scenario']==item['scenario'])
            stats = row['by_type'][item['error_type']]
            self.assertEqual(set(stats), {'injected_errors', 'detected_errors',
                                         'detected_injected_errors', 'missed_errors', 'ledr'})
            self.assertEqual(item['validator_detected'],stats['detected_injected_errors'])

    def test_unequal_injection_distribution_and_local_temporal_edits(self):
        total = Counter()
        expected = {'S1': (14,26,32), 'S2': (16,28,28), 'S3': (20,28,36)}
        for sid in ('S1','S2','S3'):
            cases = build_validator_cases(sid)
            stats = Counter(e['error_type'] for case in cases for e in case['expected_errors'])
            self.assertEqual(tuple(stats[k] for k in ('temporal','responsibility','resource')),expected[sid])
            total.update(stats)
            self.assertEqual({c['profile'] for c in cases},set(range(8)))
            for case in cases:
                kinds = Counter(e['error_type'] for e in case['expected_errors'])
                plan = VALIDATOR_INJECTION_PLAN[sid]
                variant = case['variant'] % 2
                self.assertEqual(kinds['temporal'],plan['temporal_pairs'][variant] if case['profile'] & 1 else 0)
                self.assertEqual(kinds['responsibility'],plan['responsibility_steps'][variant] if case['profile'] & 2 else 0)
                self.assertLessEqual(kinds['temporal'],4)
                if case['profile'] & 4:
                    self.assertEqual(kinds['resource'],2*plan['resource_single_tasks']+plan['resource_unknown_types']+1)
                    self.assertIn('two_task_concurrency',case['mutations'])
                    self.assertIn('unknown_resource_types',case['mutations'])
        self.assertLess(total['temporal'],total['responsibility'])
        self.assertLess(total['responsibility'],total['resource'])

    def test_three_characters_two_characters_repeats_and_conflicts(self):
        self.assertTrue(three_character_match('环境检测','环境监测'))
        self.assertFalse(three_character_match('环境保护','环境监测'))
        self.assertFalse(three_character_match('环环环','环环环'))
        validator = ResponsibilityValidator({'环境监测组':{'环境监测'}},strict=True)
        script = Script('S1',[ScriptStep(1,role='环境监测组',action='环境检测')])
        self.assertEqual(validator.validate(script),[])
        validator.conflicts = {('环境监测组','环境检测')}
        self.assertEqual(validator.validate(script)[0].error_type,'responsibility')
        self.assertEqual(semantic_match('环境检测','',{'环境监测'}),'环境监测')

    def test_empty_draft_rejected(self):
        with self.assertRaises(ValueError): steps_from_json({'steps':[]},'S1')
        with self.assertRaises(ValueError): steps_from_json({'steps':[1,2]},'S1')

    def test_in_place_corrector_cannot_mutate_shared_draft(self):
        fake = FakeClient()
        client = AuditedClient(fake)
        comps = {'client':client,'retriever':FakeRetriever(),'generator':LLMGenerator(client)}
        def mutate(corrector, script, errors, scenario):
            script.steps[0].description = '原地修改，用于测试分支隔离'
            return script
        with patch.object(audit,'OUTPUT',TEST_OUTPUT), contextlib.redirect_stdout(io.StringIO()):
            draft = experiments.prepare_shared_draft(self.cfg,'RAG','S1',comps,1)
            original = draft['script'].to_dict()
            with patch('experiments.LLMCorrector.correct',new=mutate):
                experiments.run_method(self.cfg,'ELCIG','S1',comps,draft,1)
        self.assertEqual(draft['script'].to_dict(),original)
        self.assertEqual(audit.fingerprint(draft['script']),draft['draft_hash'])

    def test_retrieval_failure_has_log(self):
        retriever = FakeRetriever()
        retriever.is_available = lambda: False
        comps = {'client':None,'generator':None,'retriever':retriever}
        with patch.object(audit,'OUTPUT',TEST_OUTPUT):
            with self.assertRaises(RuntimeError):
                experiments.prepare_shared_draft(self.cfg,'RAG','S1',comps,1)
        self.assertTrue(any('run_failure' in p.read_text(encoding='utf-8') for p in TEST_OUTPUT.rglob('trace.jsonl')))


if __name__ == '__main__':
    unittest.main()
