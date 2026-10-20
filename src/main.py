"""实验命令入口；所有数据写入项目 output 目录。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
import sys
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from dotenv import load_dotenv
from config import Config
from validator_report import console_comparison
from experiments import (ALIASES, METHOD_CONFIGS, build_components, run_ablation,
                         run_method, run_validator_effectiveness, save_results)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def _print_validator_errors(errors, title):
    names = {"temporal": "时序", "responsibility": "职责", "resource": "资源",
             "temporal_hint": "时序提示", "responsibility_hint": "职责提示"}
    print(f"\n【{title}】数量={len(errors)}")
    if not errors:
        print("  无")
    for number, error in enumerate(errors, 1):
        kind = names.get(error["error_type"], error["error_type"])
        print(f"  [{number}] 类型={kind} | 步骤={error['step_id']} | 严重度={error.get('severity', '未提供')}")
        print(f"      位置：{error.get('location', '未提供')}")
        print(f"      描述：{error.get('description', '未提供')}")
        print(f"      违反规则：{error['violated_rule']}")


def _print_validator_case(case, scenario_name, verbose=False):
    """Display the exact input and full detector output without changing the experiment."""
    hard_types = {"temporal", "responsibility", "resource"}
    hard = [e for e in case["detected_errors"] if e["error_type"] in hard_types]
    hints = [e for e in case["detected_errors"] if e["error_type"] not in hard_types]
    counts = Counter(e["error_type"] for e in hard)
    print("\n" + "=" * 80)
    print(f"样本={case['sample_id']} | 场景={case['scenario']} {scenario_name}")
    print(f"样本类型：{'含已标注错误' if case['expected_errors'] else '无错误对照'}")
    if case.get("boundary_case"):
        print("边界测试样本：", case["annotation_reason"])
        print("漏检原因：", case["miss_reason"])
    print("\n【原始样本数据（送入验证器的完整 JSON）】")
    print(json.dumps(case["script"], ensure_ascii=False, indent=2))
    print(f"\n【验证结果】{'不通过' if hard else '通过（未检出硬错误）'}")
    print(f"预期错误数量={len(case['expected_errors'])} | 检出错误数量={len(hard)} | "
          f"其中检出的注入错误={case['detected_injected_errors']} | 漏检错误数量={case['missed_errors']}")
    print(f"分类数量：时序={counts['temporal']} | 职责={counts['responsibility']} | "
          f"资源={counts['resource']} | 提示={len(hints)}")
    _print_validator_errors(hard, "验证不通过的错误详情")
    if hints:
        _print_validator_errors(hints, "验证提示（不计入不通过错误数量）")
    if case["missed_errors"]:
        print("\n【漏检的预期错误】")
        print(json.dumps(case["missed_error_details"], ensure_ascii=False, indent=2))
    if case["additional_error_details"]:
        _print_validator_errors(case["additional_error_details"], "额外检出的错误（不计入注入错误检测率）")
    if verbose:
        print("\n【完整预期错误标注】")
        print(json.dumps(case["expected_errors"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="ELCIG 共享初稿对照实验 / 人工构造验证器测试")
    parser.add_argument("--config", default=os.path.join(ROOT,"config.yaml"))
    parser.add_argument("--method", choices=list(METHOD_CONFIGS)+list(ALIASES), default="ELCIG")
    parser.add_argument("--scenario", choices=["S1","S2","S3"], default=None,
                        help="单方法默认 S1；实验模式不指定时使用配置中的所有场景")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ablation", action="store_true", help="四种方法共享两份初稿的对照实验")
    group.add_argument("--validator-test", action="store_true", help="实验1：不调用LLM或知识库")
    parser.add_argument("--repeats", type=positive, default=None, help="每个场景的独立重复次数（默认配置值或1）")
    parser.add_argument("--samples", type=positive, default=36,
                        help="实验1模板样本数，默认36；三场景默认包含228条预期错误；含边界样本后总数最多100")
    parser.add_argument("--include-boundary-case", action="store_true",
                        help="实验1每个所选场景追加1条边界样本；三场景共追加3条、3个预期错误")
    parser.add_argument("--errors", type=positive, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="实验1额外打印每条样本的完整预期错误标注")
    args = parser.parse_args()
    if args.include_boundary_case and not args.validator_test:
        parser.error("--include-boundary-case 只能与 --validator-test 一起使用")
    if args.errors is not None:
        parser.error("旧 --errors 注入计数方式已废弃；请用 --samples 36 设置独立标注样本数")
    load_dotenv(os.path.join(ROOT,".env"))
    cfg = Config.from_file(args.config)
    if args.scenario:
        cfg.experiment["scenarios"] = [args.scenario]
    repeats = args.repeats if args.repeats is not None else int(cfg.experiment.get("repeats",1))
    if repeats < 1:
        parser.error("重复次数必须大于0")
    if args.validator_test:
        if args.samples < len(cfg.experiment.scenarios)*8:
            parser.error("--samples 至少为场景数×8")
        boundary_count = len(cfg.experiment.scenarios) if args.include_boundary_case else 0
        if args.samples + boundary_count > 100:
            parser.error(f"实验1样本总数最多100条；当前边界样本{boundary_count}条，--samples 最多{100 - boundary_count}")
        eff = run_validator_effectiveness(cfg, samples=args.samples,
                                         include_boundary_case=args.include_boundary_case)
        for row in eff:
            for case in row["cases"]:
                _print_validator_case(case, row["scenario_name"], verbose=args.verbose)
            print("\n" + "=" * 80)
            print(f"【场景汇总】{row['scenario']} {row['scenario_name']}")
            print(f"{row['scenario']} 样本={row['samples']} 错误样本={row['error_samples']} "
                  f"LEDR={row['ledr']:.2%}")
            print(f"预期错误总数={row['injected_errors']} | 检出错误总数={row['detected_errors']} | "
                  f"其中检出的注入错误={row['detected_injected_errors']} | 漏检错误总数={row['missed_errors']}")
            for kind, name in (("temporal", "时序"), ("responsibility", "职责"), ("resource", "资源")):
                stats = row["by_type"][kind]
                print(f"{name}：预期={stats['injected_errors']} | 检出={stats['detected_errors']} | "
                      f"其中检出的注入错误={stats['detected_injected_errors']} | 漏检={stats['missed_errors']}")
        print("\n【实验总计】")
        print(f"样本数={sum(row['samples'] for row in eff)} | "
              f"预期错误总数={sum(row['injected_errors'] for row in eff)} | "
              f"检出错误总数={sum(row['detected_errors'] for row in eff)} | "
              f"其中检出的注入错误={sum(row['detected_injected_errors'] for row in eff)} | "
              f"漏检错误总数={sum(row['missed_errors'] for row in eff)}")
        injected = sum(row['injected_errors'] for row in eff)
        matched = sum(row['detected_injected_errors'] for row in eff)
        print(f"整体LEDR={round(matched / injected, 4):.2%}（检出的注入错误数量 / 预期错误数量）")
        print(console_comparison(eff))
        result_path = save_results(cfg,[],eff)
        print("统计表：", os.path.join(os.path.dirname(result_path), "comparison.csv"))
        print("表格预览：", os.path.join(os.path.dirname(result_path), "comparison.html"))
        print("结果：",result_path)
        return
    if args.ablation:
        results = run_ablation(cfg,repeats=repeats)
    else:
        comps = build_components(cfg)
        results = [run_method(cfg,args.method,args.scenario or "S1",comps,repeat=r)
                   for r in range(1,repeats+1)]
    for row in results:
        print(f"{row['scenario']} {row['method']} 重复{row['repeat']} "
              f"初始/最终错误={row['initial_errors']}/{row['final_errors']} "
              f"ERedR={row['eredr']} CSR={row['csr']} 修正轮次={row['iterations']}")
        print("数据：",row['output_dir'])
    print("汇总：",save_results(cfg,results,[]))


if __name__ == "__main__":
    main()
