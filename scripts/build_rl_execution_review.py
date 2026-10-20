"""Publish a source-linked research review, separate from experiment results."""
from pathlib import Path
import html
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]
from build_bottom_drawer_investigation_report import CSS


def link(url, text):
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(text)}</a>'


def main():
    data = json.loads((ROOT / 'docs/rl_execution_sources.json').read_text())
    out = ROOT / 'results/rl-execution-literature-review'; out.mkdir(exist_ok=True)
    papers = data['papers']; cards = []
    for group in dict.fromkeys(p['group'] for p in papers):
        cards.append(f'<h2>{html.escape(group)}</h2>')
        for p in [p for p in papers if p['group'] == group]:
            links = [link(p['paper'], 'Paper')]
            for key, label in [('primary_detail', 'Primary detail'), ('project', 'Author project / video'), ('code', 'Code')]:
                if key in p: links.append(link(p[key], label))
            cards.append(f'''<article class="card paper" id="{p['id']}"><span class="pill">{html.escape(p['priority'])}</span>
<h3>{link(p['paper'], p['title'])}</h3><p class="muted">{html.escape(p['authors'])} · {html.escape(p['venue'])}</p>
<p><b>Mechanism.</b> {html.escape(p['method'])}</p><p><b>Evidence.</b> {html.escape(p['evidence'])}</p>
<p><b>Relevance and limit.</b> {html.escape(p['fit'])}</p><p><b>Training / demonstrations.</b> {html.escape(p['human_data'])}</p>
<p class="muted">{html.escape(p['code_note'])}</p><nav>{' · '.join(links)}</nav></article>''')
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reactive execution around a planned trajectory — literature review</title><style>{CSS}
.paper p{{margin:10px 0}}.equation{{padding:20px;background:#e8f2f5;font-size:20px;text-align:center;border-radius:10px}}.architecture{{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap}}.architecture div{{padding:16px;background:#e8f2f5;border-radius:8px;flex:1;min-width:170px}}input{{width:100%;font:inherit;padding:12px;border:1px solid #a7bdc9;border-radius:8px}}[hidden]{{display:none!important}}
</style></head><body><main><div class="eyebrow">Research review · Primary sources · {data['review_date']}</div>
<h1>Planned motion with learned, reactive execution is an established research direction</h1>
<p class="lead">The closest terms are <b>residual reinforcement learning</b>, <b>reference correction</b>, <b>variable impedance learning</b> and <b>adaptive contact control</b>. A nominal trajectory can guide execution while a fast feedback loop corrects what actually happens.</p>
<nav><a href="#distinctions">Which error are we correcting?</a><a href="#closest">Closest work</a><a href="#proposal">Application to our pipeline</a><a href="#library">16-paper review</a><a href="../bottom-drawer-investigation-report/index.html">Bottom-drawer evidence</a></nav>
<p class="notice good">This is a selective literature review, not an exhaustive systematic search. Primary papers, author pages and linked repositories were checked. No RL method was trained or reproduced in this work. Our new drawer experiment uses a simple measured-reference update.</p>
<h2 id="distinctions">Three different errors need different corrections</h2>
<table><thead><tr><th>Failure</th><th>Appropriate response</th><th>Relevant prior</th></tr></thead><tbody>
<tr><td>The trajectory is feasible, but the real hand lags or deflects.</td><td>Modify the command so the actual robot tracks the desired path.</td><td><a href="#reference-correction">Reference correction</a>; <a href="#industreal">IndustReal</a></td></tr>
<tr><td>The estimated hinge is wrong, so the desired path conflicts with the physical door.</td><td>Adapt the path/direction or compliance while maintaining useful contact and progress.</td><td><a href="#open-sesame">Open Sesame</a>; <a href="#vices">VICES</a>; <a href="#residual-policy">residual policies</a></td></tr>
<tr><td>The target is on the table, or a bulky pinch posture cannot fit.</td><td>Repair grounding or choose a different contact mode before local tracking.</td><td><a href="../bottom-drawer-investigation-report/index.html">Our drawer diagnosis</a>; local correction alone cannot identify the intended object reliably.</td></tr>
</tbody></table>
<p><b>Real-time reaction and online learning are different.</b> A trained policy can react at every control step while its weights remain fixed. An adaptation module can infer changing dynamics from history without gradient updates. Other methods update policies between trials or use human corrections. These should not be reported as the same capability.</p>
<h2 id="closest">The most useful starting points</h2>
<table><thead><tr><th>Question</th><th>Start here</th><th>What to take from it</th></tr></thead><tbody>
<tr><td>Can a learner correct an externally supplied trajectory?</td><td><a href="#reference-correction">Pavlichenko &amp; Behnke, 2022</a></td><td>Reference-conditioned correction outside an existing joint controller.</td></tr>
<tr><td>Can we keep a conventional controller and learn only its missing part?</td><td><a href="#residual-rl">Residual RL, 2019</a>; <a href="#residual-policy">Residual Policy Learning, 2018</a></td><td>A base policy plus a bounded, learned adjustment.</td></tr>
<tr><td>How should a door controller tolerate an incorrect hinge estimate?</td><td><a href="#open-sesame">Open Sesame, 2012</a>; <a href="#vices">VICES, 2019</a></td><td>Adaptive force/velocity and adjustable compliance.</td></tr>
<tr><td>How can real contacts differ from simulation?</td><td><a href="#industreal">IndustReal, 2023</a>; <a href="#forge">FORGE, 2025</a></td><td>Deployment action semantics, force observations and randomized dynamics.</td></tr>
<tr><td>Has language + planning + RL already been combined?</td><td><a href="#psl">Plan-Seq-Learn, 2024</a>; <a href="#mopa">MoPA-RL, 2020</a></td><td>Important system-level baselines for any novelty claim.</td></tr>
<tr><td>What if we later use a demonstration or human correction?</td><td><a href="#rlfd">rLfD, 2022</a>; <a href="#transic">TRANSIC, 2024</a></td><td>Useful future variants with a different supervision budget.</td></tr>
</tbody></table>
<p>The Baxter reference-correction study reports a <b>20 Hz</b> learned outer loop. VICES also reports a <b>20 Hz</b> policy over a faster <b>500 Hz</b> low-level controller. These are concrete feasibility examples, not timing measurements of our implementation. <a href="https://www.ais.uni-bonn.de/papers/ICRA_2022_Pavlichenko.pdf">Baxter paper</a> · <a href="https://arxiv.org/pdf/1906.08880">VICES paper</a></p>
<h2 id="proposal">A concrete adaptation for our VLM + planner system</h2>
<p>The following is a design recommendation inferred from the literature and our diagnosis; it is not a result reported by one paper.</p>
<div class="architecture"><div><b>VLM + RGB-D</b><br>Target, contact mode, approximate articulation</div><div><b>Motion planner</b><br>Approach and nominal local path</div><div><b>Reactive controller</b><br>Measured motion, contact-aware offset and timing</div><div><b>Robot / physics</b><br>Actual motion feeds the controller and estimator</div></div>
<p>Keep the planner responsible for nominal reachability, approach and obstacle geometry. Let a lightweight controller observe the actual hand, gripper, tracked handle and a short future window of the nominal path. It can output a small pose correction and, if the interface supports it, compliance and progress speed. Luna stays outside the fast control loop.</p>
<div class="equation">commanded pose = planned pose ∘ learned small pose correction</div>
<p>For implementation, compose rigid transforms or exponential-map twists; do not add quaternion coordinates. Express corrections in the contact frame so a shared rule can transfer between drawers and door orientations. For an inaccurate hinge, reward actual task progress and retained contact while penalizing excessive force and abrupt motion. Exact reference tracking should be a soft objective when the reference is uncertain.</p>
<p>Use robot proprioception, RGB-D tracking and a hardware-realizable wrench estimate/sensor as actor inputs. Native object labels, true articulation state and per-geometry contact forces stay outside the deployed policy. If simulator truth is used for training rewards or a privileged critic, document that separately from actor observations and evaluation-only diagnostics.</p>
<h3>First compare feedback designs before adding training</h3>
<ol><li><b>Unmodified nominal reference:</b> the existing joint servo and fixed planned pull.</li><li><b>Measured-state reference update:</b> the current diagnostic correction, with its manual contact-mode assumptions explicit.</li><li><b>Classical compliance/adaptive contact:</b> a force/velocity or admittance baseline for hinge and surface error.</li><li><b>Learned residual:</b> same planner and observations, with randomized hinge estimates, friction, contact offsets, handle sizes and delays during training.</li></ol>
<p>Freeze the controller before evaluation. Test unseen physical geometries and families separately from new initial states of the same asset. Measure native task success, retained contact, peak and integrated force, unintended object motion, completion time, sample cost and policy latency. Keep perception failures in the denominator. No guarantee of completion or collision-free contact follows from adding RL.</p>
<h3>What the bottom drawer already tells us</h3>
<p>The measured-hand variant succeeded in four development opening cases without RL. It changes the reference once after contact, then executes the complete pull. Large approach loads remain. This gives us a useful simple baseline: any learned controller should improve success or contact quality beyond that baseline, rather than receiving credit merely for adjusting the reference.</p>
<h3>How AlphaNIST fits</h3>
<p><a href="https://alpha-nist.github.io/">AlphaNIST</a> and its linked <a href="https://sgs-rl.github.io/">SGS approach</a> are relevant to simulation training, curricula and visual-policy transfer. They are not an implementation of our planner-conditioned residual controller. A useful adaptation would retain our nominal plan as policy conditioning and train corrections across controlled model errors. That is a proposed experiment, not a capability established by their headline assembly success rate.</p>
<p><b>Research positioning:</b> combining planning and learned feedback is already well established. A defensible contribution would need evidence about our particular language-grounded contact representation, uncertainty handling or transfer across articulated objects, compared against these existing control and planning/RL baselines. Demonstration-free evaluation still permits substantial RL training; it should not be described as training-free.</p>
<h2 id="library">Paper-by-paper review</h2><p><label for="filter">Filter by title, topic, robot or method</label></p><input id="filter" type="search" placeholder="For example: reference, hinge, demonstration, Panda, code">
{''.join(cards)}
<p class="muted">Machine-readable bibliography: <a href="references.json">references.json</a>. Method and evidence statements are source-based; relevance assessments and our proposed experiment are interpretations. Code availability is not a compatibility or reproducibility guarantee.</p>
</main><script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase().trim();document.querySelectorAll('.paper').forEach(p=>p.hidden=!!q&&!p.textContent.toLowerCase().includes(q));}});</script></body></html>'''
    (out / 'index.html').write_text(document)
    (out / 'references.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'review': str(out/'index.html'), 'papers': len(papers)}))


if __name__ == '__main__':
    main()
