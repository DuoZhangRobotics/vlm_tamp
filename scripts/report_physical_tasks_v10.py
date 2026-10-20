"""Contact-policy report including the selected plate's observed rim."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from report_physical_tasks_v7 import collect
from report_physical_tasks_v8 import render
from try_rekep_bowl_v3 import save

ROOT = Path(__file__).resolve().parents[1]


def support_plot(folder, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    seed = np.load(folder / 'target-cloud.npy', allow_pickle=False)
    target = np.load(folder / 'observed-target-contact-cloud.npy', allow_pickle=False)
    centre = seed.mean(0)
    with np.load(folder / 'initial-scene.npz') as scene:
        static = scene['static_points']
    near = static[np.linalg.norm(static[:, :2] - centre[:2], axis=1) < .10]
    near = near[::max(1, len(near) // 12000)]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for ax, dims, labels in zip(axes, [(0, 1), (0, 2)], [('World x (m)', 'World y (m)'), ('World x (m)', 'Height (m)')]):
        ax.scatter(near[:, dims[0]], near[:, dims[1]], c='#aab4c0', alpha=.25, s=1, label='Other measured surfaces')
        ax.scatter(target[:, dims[0]], target[:, dims[1]], c='#d48b17', s=2, label='Connected plate support')
        ax.scatter(seed[:, dims[0]], seed[:, dims[1]], c='#2575ad', s=2, label='Original RGB plate seed')
        ax.set(xlabel=labels[0], ylabel=labels[1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7, markerscale=3)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/physical-tasks-v10-eval')
    parser.add_argument('--output', default='results/task-grounding-contact-v10-report')
    args = parser.parse_args()
    base, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve reports')
    output.mkdir(parents=True)
    data = collect(base)
    data['selected_support_surface_expansion_enabled'] = True
    save(output / 'evaluation.json', data)
    for name in ['protocol.json', 'configuration.json']:
        shutil.copyfile(base / name, output / name)
    render(data, output)
    path = output / 'index.html'
    document = path.read_text().replace('Correcting task contact permissions', 'Contact permissions and the selected plate rim')
    document = document.replace('Initial support while lifting; selected plate while placing/releasing. Other obstacles stay checked.',
        'Initial support while lifting; the selected plate including its connected observed rim while placing/releasing. The table and disconnected objects stay checked.')
    document = document.replace('<h2>Actual LIBERO success</h2>', '<p>The initial RGB plate mask covered only the flat center. '
        'The checker also needs to recognize the raised rim as part of the selected support. This version grows that RGB-D seed '
        'through connected measured surfaces in a bounded 2.5 cm lateral region above the tabletop. '
        'No simulator identities or plate CAD are used. Each bowl episode includes a plot of the seed and expanded support below.</p><h2>Actual LIBERO success</h2>')
    for row in data['episodes']:
        if row['task_index'] != 8 or row['status'] == 'pending' or not (ROOT / row['folder'] / 'observed-target-contact-cloud.npy').is_file():
            continue
        assets = output / f"goal08-state{row['initial_state']:03d}"
        support_plot(ROOT / row['folder'], assets / 'support-contact.png')
        marker = f"<article><h2>Bowl onto plate · state {row['initial_state']}:"
        start = document.index(marker)
        end = document.index('</article>', start)
        caption = (f'<figure><img class="scene" src="{assets.name}/support-contact.png" alt="Original plate seed and connected rim support">'
                   '<figcaption>Blue: original plate-center seed. Orange: connected observed plate support. '
                   'Gray: surrounding observed geometry, including the excluded tabletop.</figcaption></figure>')
        document = document[:end] + caption + document[end:]
    document = document.replace('<h2>Actual LIBERO success</h2>', '<p>The moving bowl cloud is bounded by its RGB-selected height range and an observed local tabletop upper band. Table points stay stationary. Current connected source ownership prevents newly visible bowl surfaces from becoming stationary ghosts.</p><h2>Actual LIBERO success</h2>')
    path.write_text(document)
    print(json.dumps({key: value for key, value in data.items() if key not in ['episodes', 'source_freeze_checks']}))


if __name__ == '__main__':
    main()
