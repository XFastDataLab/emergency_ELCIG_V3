"""Scenario/type exports of validator results."""
import csv
import html
from pathlib import Path
from audit import write_json

TYPES = (("temporal", "时序错误"), ("responsibility", "职责错误"), ("resource", "资源错误"))
SCENARIO_NAMES = {"S1": "危化品泄露", "S2": "台风灾害", "S3": "地质灾害"}
HEADERS = ["场景", "错误类型", "错误样本个数（错误条目）", "验证器检测数", "验证器检测率"]


def build_comparison_report(results):
    rows = []
    for result in results:
        for kind, name in TYPES:
            stats = result["by_type"][kind]
            # The numerator contains only detections matched to frozen gold labels.
            rows.append({"scenario": result["scenario"], "scenario_name": SCENARIO_NAMES[result["scenario"]],
                         "error_type": kind, "error_type_name": name,
                         "error_instances": stats["injected_errors"],
                         "validator_detected": stats["detected_injected_errors"],
                         "validator_rate": stats["ledr"]})
    injected = sum(row["error_instances"] for row in rows)
    detected = sum(row["validator_detected"] for row in rows)
    total = {"scenario": "ALL", "scenario_name": "总计", "error_type": "all", "error_type_name": "—",
             "error_instances": injected, "validator_detected": detected,
             "validator_rate": round(detected / injected, 4) if injected else None}
    boundaries = [{"sample_id": case["sample_id"], "scenario": case["scenario"],
                   "expected_errors": case["expected_errors"], "annotation_reason": case["annotation_reason"],
                   "miss_reason": case["miss_reason"], "missed_errors": case["missed_errors"],
                   "detected_errors": case["detected_errors"]}
                  for result in results for case in result["cases"] if case.get("boundary_case")]
    return {"title": "分场景、分错误类型验证器检错统计",
            "note": "错误样本个数按违反约束的错误条目统计，非脚本样本数；检测率=检出的注入错误/预期错误。",
            "script_samples": sum(result["samples"] for result in results),
            "rows": rows, "total": total, "boundary_explanations": boundaries}


def _percentage(value):
    return f"{value:.2%}" if value is not None else "不适用"


def _display_row(row):
    return [row["scenario_name"], row["error_type_name"], row["error_instances"], row["validator_detected"],
            _percentage(row["validator_rate"])]


def console_comparison(results):
    report = build_comparison_report(results)
    lines = ["\n【" + report["title"] + "】", "\t".join(HEADERS)]
    lines.extend("\t".join(map(str, _display_row(row))) for row in report["rows"] + [report["total"]])
    lines.append(report["note"])
    return "\n".join(lines)


def save_comparison_report(folder, results):
    folder = Path(folder)
    report = build_comparison_report(results)
    write_json(folder / "comparison.json", report)
    write_json(folder / "boundary_explanations.json", report["boundary_explanations"])
    with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows(_display_row(row) for row in report["rows"] + [report["total"]])
    cells = []
    for i, row in enumerate(report["rows"]):
        values = _display_row(row)
        scenario_cell = f'<td rowspan="3">{html.escape(values[0])}</td>' if i % 3 == 0 else ""
        cells.append("<tr>" + scenario_cell + "".join(f"<td>{html.escape(str(v))}</td>" for v in values[1:]) + "</tr>")
    cells.append('<tr class="total">' + "".join(f"<td>{html.escape(str(v))}</td>" for v in _display_row(report["total"])) + "</tr>")
    reasons = "".join(f"<li><b>{html.escape(b['sample_id'])}</b>：{html.escape(b['annotation_reason'])}<br>漏检原因：{html.escape(b['miss_reason'])}</li>" for b in report["boundary_explanations"])
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>验证器检错对比</title>'
            '<style>body{font:16px "Microsoft YaHei",sans-serif;margin:32px;color:#111}table{border-collapse:collapse;width:1100px;max-width:100%}th,td{border:1px solid #333;padding:12px 10px;text-align:center}th,.total{font-weight:bold}li{margin:16px 0;line-height:1.7}p{line-height:1.7}</style>'
            f'<h2>{html.escape(report["title"])}</h2><p>{html.escape(report["note"])}</p>'
            f'<table><thead><tr>{"".join(f"<th>{html.escape(h)}</th>" for h in HEADERS)}</tr></thead><tbody>{"".join(cells)}</tbody></table>'
            f'<p>脚本样本数：{report["script_samples"]}。</p>'
            f'<h3>边界样本漏检原因</h3><ul>{reasons}</ul></html>')
    (folder / "comparison.html").write_text(page, encoding="utf-8")
    return report
