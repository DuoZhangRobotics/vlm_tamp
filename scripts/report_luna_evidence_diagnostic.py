"""Build a local, inspectable report for the two-stage Luna diagnostic."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import statistics


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def e(value):
    return html.escape(str(value))


def link(path, label):
    return f'<a href="{e(path)}">{e(label)}</a>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    out = Path(parser.parse_args().output)
    cases = read(out / 'cases-private.json')
    jobs = read(out / 'schedule-private.json')
    status = read(out / 'status.json', {})
    rows = [read(out / 'responses' / job['run_id'] / 'result.json') for job in jobs]
    rows = [r for r in rows if r]
    aggregates = {}
    for condition in ['baseline', 'predict-inspect-revise']:
        group = [r for r in rows if r['condition'] == condition]
        check_counts = Counter(k for r in group for k, v in r.get('score', {}).get('checks', {}).items() if v)
        usage = Counter()
        for r in group:
            for stage in ['before', 'after']:
                for u in r.get(stage, {}).get('usage_records', []):
                    usage.update(u)
        aggregates[condition] = {
            'completed_sessions': len(group), 'valid_decisions': sum(r['valid_decision'] for r in group),
            'all_label_checks_passed': sum(r.get('score', {}).get('all_label_checks_passed', False) for r in group),
            'checks_passed': dict(check_counts),
            'mean_model_session_seconds': statistics.mean(sum(r.get(s, {}).get('wall_seconds', 0) for s in ['before', 'after']) for r in group) if group else None,
            'mean_observed_command_seconds': statistics.mean(sum(r.get(s, {}).get('observed_command_seconds', 0) for s in ['before', 'after']) for r in group) if group else None,
            'mean_outside_observed_commands_seconds': statistics.mean(sum(r.get(s, {}).get('outside_observed_commands_seconds', 0) for s in ['before', 'after']) for r in group) if group else None,
            'raw_cli_usage': dict(usage),
        }
    pairs = []
    for case in cases:
        for repeat in [1, 2]:
            matched = {r['condition']: r for r in rows if r['case_id'] == case['case_id'] and r['repeat'] == repeat}
            if len(matched) == 2:
                pairs.append({'case_id': case['case_id'], 'repeat': repeat,
                    **{k: v.get('score', {}).get('all_label_checks_passed', False) for k, v in matched.items()}})
    summary = {'status': status, 'conditions': aggregates, 'pairs': pairs,
               'limits': 'Structured labels are screening metrics; inspect supporting evidence and next steps. Reused correlated checkpoints, not physical success or transfer.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'timing.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'case', 'condition', 'repeat', 'before_seconds', 'after_seconds', 'observed_commands_seconds', 'outside_observed_commands_seconds', 'preflight_seconds', 'archive_seconds', 'total_seconds'])
        for r in rows:
            writer.writerow([r['run_id'], r['case_id'], r['condition'], r['repeat'],
                             r.get('before', {}).get('wall_seconds'), r.get('after', {}).get('wall_seconds'),
                             sum(r.get(s, {}).get('observed_command_seconds', 0) for s in ['before', 'after']),
                             sum(r.get(s, {}).get('outside_observed_commands_seconds', 0) for s in ['before', 'after']),
                             r.get('preflight_seconds'), r.get('archive_seconds'), r['total_seconds']])
    document = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Luna evidence loop diagnostic</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#15243a;background:#f5f7fa}a{color:#0059aa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border:1px solid #cad3df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff2f7;padding:14px;font-size:13px}details{margin:16px 0;background:white;padding:12px;border-radius:8px}summary{cursor:pointer;font-weight:650}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.images img{width:100%}.label{font-size:13px}.muted{color:#526477}.good{color:#12633d}.bad{color:#9c2f22}</style>',
        '<h1>Luna: predict → inspect → revise</h1>',
        f'<p><b>{e(status.get("phase"))}</b> · {len(rows)}/{len(jobs)} sessions · six recorded checkpoints · two repetitions per condition</p>',
        '<p>Both conditions receive identical original before/after images directly. The intervention asks for an observable prediction, then an evidence-based revision. No live simulator, stronger teacher, cross-case memory update or task-specific answer.</p>',
        '<p>' + ' · '.join(link(p, label) for p, label in [('protocol.json', 'Registered protocol'), ('rubric-private.json', 'Private rubric'), ('cases-private.json', 'Provenance and native evidence'), ('summary.json', 'Machine-readable summary'), ('timing.csv', 'Timing CSV')]) + '</p>',
        '<h2>Current structured-label results</h2><p class="muted">These checks do not alone establish sound reasoning or feasible recovery. This is a selected replay diagnostic, not a physical success evaluation.</p>',
        '<table><tr><th>Condition</th><th>Valid outputs</th><th>All registered label checks</th><th>Mean two-turn wall time</th></tr>']
    for name, data in aggregates.items():
        seconds = data['mean_model_session_seconds']
        document.append(f'<tr><td>{e(name)}</td><td>{data["valid_decisions"]}/{data["completed_sessions"]}</td><td>{data["all_label_checks_passed"]}/{data["completed_sessions"]}</td><td>{f"{seconds:.1f} s" if seconds is not None else "—"}</td></tr>')
    document.append('</table><p class="muted">Wall time includes model inference, local analysis tools, scheduling and transport. Planning, grasp search, IK and physical execution are all zero in this offline diagnostic. CLI token totals include repeated/cached context; they are not billing or peak context.</p>')
    review = read(out / 'review-private.json')
    if review:
        document.append('<h2>Evidence and next-step review</h2><pre>' + e(json.dumps(review, indent=2)) + '</pre>')
    for case in cases:
        key = case['case_id']
        document.append(f'<h2>{e(key)} — {e(case["private_category"])}</h2><p>Observer labels shown here only; actors did not receive case categories or native measurements. Source: {e(case["source_trial"])} event {case["event"]}.</p>')
        document.append('<details><summary>Original before/after images and recorded request/result</summary><div class="images">')
        for i, camera in enumerate(['frontview', 'agentview', 'wrist', 'arm_side']):
            for stage in ['before', 'after']:
                path = f'packets/{key}/{stage}/' + case['stages'][stage]['current_images'][i]
                document.append(f'<div><span class="label">{e(stage)} {camera}</span><a href="{e(path)}"><img loading="lazy" src="{e(path)}" alt="{e(stage)} {camera}"></a></div>')
        document.append('</div><p>' + link(f'packets/{key}/before/pending-request.json', 'Recorded request') + ' · ' + link(f'packets/{key}/after/latest-result.json', 'Original public result') + ' · ' + link(f'packets/{key}/after/history-compact.jsonl', 'Compact history') + '</p></details>')
        for r in [r for r in rows if r['case_id'] == key]:
            rel = f'responses/{r["run_id"]}'
            before = read(out / rel / 'before.json', {})
            after = read(out / rel / 'decision.json', {})
            passed = r.get('score', {}).get('all_label_checks_passed', False)
            document.append(f'<details><summary>{e(r["run_id"])} · {e(r["condition"])} · repetition {r["repeat"]} · label screen {"PASS" if passed else "FAIL"}</summary>')
            document.append('<p>' + ' · '.join(link(f'{rel}/{p}', label) for p, label in [('before-prompt.txt', 'Before input'), ('after-prompt.txt', 'After input'), ('before-events.jsonl', 'Before tool I/O'), ('after-events.jsonl', 'After tool I/O'), ('result.json', 'Timing / checks')]) + '</p>')
            document.append('<h3>Before note</h3><pre>' + e(json.dumps(before, indent=2)) + '</pre><h3>Observed result and decision</h3><pre>' + e(json.dumps(after, indent=2)) + '</pre>')
            document.append('<h3>Screening checks</h3><pre>' + e(json.dumps(r.get('score', {}), indent=2)) + '</pre></details>')
    document.append('</html>')
    temporary = out / 'watch.html.tmp'
    temporary.write_text('\n'.join(document))
    temporary.replace(out / 'watch.html')
    print(json.dumps({'completed_sessions': len(rows), 'report': str(out / 'watch.html')}))


if __name__ == '__main__':
    main()
