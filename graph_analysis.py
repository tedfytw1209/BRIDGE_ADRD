import os
import csv
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from pyvis.network import Network

import config
from data_loader import get_or_create_split, fit_global_scalers, apply_global_scalers, load_global_scalers, PatientDataset
from graph_builder import MultiModalGraphConstructor
from gcn import PatientGCNEncoder
from fusion import ModalityFusion
from decoders import ModalityDecoder
from utils import set_deterministic_seed


LABEL_NAMES = {0: 'Normal', 1: 'MCI', 2: 'Dementia'}
DIAGNOSIS_COLORS = {'Normal': '#2ecc71', 'MCI': '#f39c12', 'Dementia': '#e74c3c'}
MATCH_COLORS = {'SAME': '#3498db', 'DIFF': '#bdc3c7'}


def load_model(checkpoint_path, modality_dims, device):
    graph_builder = MultiModalGraphConstructor(
        modality_dims, hidden_dim=config.config['gcn_dim'], num_heads=config.config['graph_heads'],
        theta=config.config['theta'], top_k=config.config['top_k'], dropout=config.config['dropout'], aggregate=True
    ).to(device)
    gcn_encoder = PatientGCNEncoder(hidden_dim=config.config['gcn_dim'], num_layers=config.config['num_gcn_layers'], dropout=config.config['dropout']).to(device)
    fusion = ModalityFusion(hidden_dim=config.config['gcn_dim']).to(device)
    decoders = {m: ModalityDecoder(config.config['gcn_dim'], modality_dims[m]).to(device)
                for m in config.config['modalities_to_impute']}

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    graph_builder.load_state_dict(ckpt['graph_builder'])
    gcn_encoder.load_state_dict(ckpt['gcn_encoder'])
    fusion.load_state_dict(ckpt['fusion'])
    for m, d in decoders.items():
        d.load_state_dict(ckpt['decoders'][m])

    graph_builder.eval(); gcn_encoder.eval(); fusion.eval()
    for d in decoders.values(): d.eval()

    fold_idx = ckpt.get('fold_idx', '?')
    epoch = ckpt.get('epoch', '?')
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch {epoch}, fold {fold_idx})")
    return graph_builder, fold_idx, epoch


@torch.no_grad()
def get_graphs_for_batch(graph_builder, batch, device):
    labels = batch['label'].detach().cpu().numpy() if torch.is_tensor(batch['label']) else np.array(batch['label'])
    pids = batch['patient_id'].detach().cpu().numpy() if torch.is_tensor(batch['patient_id']) else np.array(batch['patient_id'])

    available = sorted(f'm{i}' for i in range(1, 8) if f'm{i}' in batch)
    x_dict = {m: batch[m].to(device) for m in available}
    feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in available}
    x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in available}
    patient_mask_dict = {m: (feat_mask_dict[m].sum(dim=1) > 0).float() for m in available}

    A_hat, _ = graph_builder(x_dict_filled, feat_mask_dict=feat_mask_dict, patient_mask_dict=patient_mask_dict)
    patient_presence = {m: patient_mask_dict[m].detach().cpu().numpy().astype(bool) for m in available}
    return A_hat.cpu().numpy(), labels, pids, patient_presence


def save_html(per_patient, A_hat, labels, pids, patient_presence, modality_order, out_dir):
    net = Network(height='100vh', width='100%', directed=True, bgcolor='#1a1a2e', font_color='white')
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=160, spring_strength=0.04, damping=0.09)

    all_w = [float(A_hat[p['i'], j]) for p in per_patient for j in p['valid_j']]
    w_min, w_max = (min(all_w), max(all_w)) if all_w else (0.0, 1.0)
    w_range = w_max - w_min if w_max > w_min else 1.0

    # homophily lookup by batch index
    hom_by_idx = {p['i']: p['homophily'] for p in per_patient}
    added = set()

    def add_node(idx):
        nid = str(pids[idx])
        if nid in added:
            return
        added.add(nid)
        diagnosis = LABEL_NAMES.get(int(labels[idx]), 'Unknown')
        missing = ', '.join(m.upper() for m in modality_order if not patient_presence[m][idx]) or 'None'
        hom = hom_by_idx.get(idx)
        hom_line = f"\nHomophily: {hom:.3f}" if hom is not None else ''
        tooltip = (
            f"Patient ID: {nid}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Missing mods: {missing}"
            f"{hom_line}"
        )
        net.add_node(nid, label=nid, color=DIAGNOSIS_COLORS.get(diagnosis, '#888888'),
                     title=tooltip, size=22, borderWidth=2, borderWidthSelected=4)

    for p in per_patient:
        add_node(p['i'])
        for j in p['valid_j']:
            add_node(j)
            w = float(A_hat[p['i'], j])
            match = 'SAME' if labels[j] == p['label'] else 'DIFF'
            width = 1.0 + 5.0 * (w - w_min) / w_range
            net.add_edge(
                str(pids[p['i']]), str(pids[j]),
                value=width,
                color=MATCH_COLORS.get(match, '#888888'),
                title=f"Weight: {w:.4f} | {match}",
                arrows='to',
            )

    html_path = os.path.join(out_dir, 'patient_graph.html')
    try:
        net.write_html(html_path, notebook=False)
    except TypeError:
        net.write_html(html_path)

    legend = """
<div style="position:fixed;top:14px;right:14px;background:rgba(30,30,60,0.92);
            padding:14px 18px;border-radius:10px;color:white;font-family:Arial;
            font-size:13px;z-index:9999;border:1px solid #555;line-height:1.8">
  <b>Node - Diagnosis</b><br>
  <span style="color:#2ecc71">&#9632;</span> Normal &nbsp;
  <span style="color:#f39c12">&#9632;</span> MCI &nbsp;
  <span style="color:#e74c3c">&#9632;</span> Dementia
  <br><br>
  <b>Edge - Match</b><br>
  <span style="color:#3498db">&#9632;</span> SAME diagnosis<br>
  <span style="color:#bdc3c7">&#9632;</span> DIFF diagnosis<br><br>
  <i style="font-size:11px">Hover node/edge for details</i>
</div>"""
    fullscreen_css = '<style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}.vis-tooltip{white-space:pre;font-family:Arial,sans-serif;font-size:13px;line-height:1.6;}</style>'
    with open(html_path, 'r') as f:
        html = f.read()
    with open(html_path, 'w') as f:
        f.write(html.replace('</head>', fullscreen_css + '\n</head>').replace('</body>', legend + '\n</body>'))

    print(f"Saved: {html_path}")


def main():
    OUT = config.config.get('output_dir', 'output')
    out_dir = os.path.join(OUT, 'graph_analysis')
    os.makedirs(out_dir, exist_ok=True)
    set_deterministic_seed(config.config['random_state'])

    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f"Device: {device}")

    train_data, _, modality_dims = get_or_create_split(config.config, save=True)

    from data_loader import _get_paths
    scaler_path = _get_paths()['scaler_path']
    if os.path.exists(scaler_path):
        scalers, medians = load_global_scalers(scaler_path)
        print(f"Loaded scalers from {scaler_path}")
    else:
        scalers, medians = fit_global_scalers(train_data, modalities_to_scale=config.config['modalities_to_scale'])

    scaled = apply_global_scalers(train_data, scalers, medians, modalities_to_scale=config.config['modalities_to_scale'])
    train_data = {**train_data, **scaled}

    graph_builder, fold_idx, epoch = load_model(config.config['save_model_path'], modality_dims, device)

    modality_keys = sorted(k for k in train_data if k.startswith('m'))
    ds = PatientDataset({k: train_data[k] for k in modality_keys}, train_data['labels'], train_data['patient_ids'])
    loader = DataLoader(ds, batch_size=config.config['batch_size'], shuffle=False)
    batch = next(iter(loader))

    print(f"\nForward pass on first batch (N={config.config['batch_size']}) ...")
    A_hat, labels, pids, patient_presence = get_graphs_for_batch(graph_builder, batch, device)

    label_dist = {LABEL_NAMES[l]: int((labels == l).sum()) for l in [0, 1, 2]}
    print(f"Label distribution: {label_dist}")
    print(f"A_hat: {A_hat.shape}, non-zero edges={int((A_hat > 0).sum())}, mean_w={A_hat[A_hat > 0].mean():.4f}")

    top_k = min(config.config['top_k'], len(labels) - 1)
    modality_order = sorted(patient_presence.keys())

    per_patient = []
    for i in range(len(labels)):
        row = A_hat[i].copy()
        row[i] = 0.0
        top_j = np.argsort(row)[::-1][:top_k]
        valid_j = [j for j in top_j if row[j] > 0]
        same = sum(1 for j in valid_j if labels[j] == labels[i])
        homophily = same / len(valid_j) if valid_j else None
        per_patient.append({
            'i': i, 'pid': pids[i], 'label': labels[i], 'top_j': top_j, 'valid_j': valid_j, 'homophily': homophily,
            'present': [m for m in modality_order if patient_presence[m][i]],
            'missing': [m for m in modality_order if not patient_presence[m][i]],
        })

    all_weights = A_hat[A_hat > 0].ravel()
    n_isolated = sum(1 for p in per_patient if len(p['valid_j']) == 0)
    homophily_vals = [p['homophily'] for p in per_patient if p['homophily'] is not None]
    mean_homophily = np.mean(homophily_vals)
    n_gt50 = sum(1 for h in homophily_vals if h > 0.5)

    label_counts = np.array([int((labels == l).sum()) for l in [0, 1, 2]])
    random_homophily = np.sum((label_counts / len(labels)) ** 2)

    miss_pattern_counts = Counter(tuple(p['missing']) for p in per_patient)
    isolated_patients = [p for p in per_patient if len(p['valid_j']) == 0]

    summary = [
        "\nGRAPH ANALYSIS SUMMARY",
        f"Checkpoint: {config.config['save_model_path']} | fold={fold_idx} | epoch={epoch}",
        "=" * 100, "", "BATCH OVERVIEW", "-" * 60, f"  Total patients   : {len(labels)}",
        f"  Normal           : {label_dist['Normal']}", f"  MCI              : {label_dist['MCI']}",
        f"  Dementia         : {label_dist['Dementia']}", f"  Top-K used       : {top_k}", "",
        "EDGE WEIGHT STATISTICS (A_hat, non-zero edges only)", "-" * 60, f"  Non-zero edges   : {len(all_weights)}",
        f"  Min weight       : {all_weights.min():.4f}", f"  Max weight       : {all_weights.max():.4f}",
        f"  Mean weight      : {all_weights.mean():.4f}", f"  Median weight    : {np.median(all_weights):.4f}", "",
        "HOMOPHILY (fraction of top-K neighbors with same diagnosis)", "-" * 60,
        f"  Mean homophily         : {mean_homophily:.3f}",
        f"  Random baseline        : {random_homophily:.3f}  (expected by label prevalence alone)",
        f"  Lift over random       : {mean_homophily - random_homophily:+.3f}",
        f"  Patients with >50% SAME neighbors : {n_gt50} / {len(labels)}", "",
        "ISOLATED PATIENTS (zero positive-weight neighbors after top-K filtering)", "-" * 60,
        f"  Count: {n_isolated}",
    ]
    for p in isolated_patients:
        miss_str = ", ".join(m.upper() for m in p['missing']) if p['missing'] else "None"
        summary.append(f"    ID={p['pid']} | Label={LABEL_NAMES[p['label']]} | Missing: {miss_str}")

    summary += ["", "MODALITY MISSINGNESS PATTERNS IN THIS BATCH", "-" * 60]
    for pattern, count in sorted(miss_pattern_counts.items(), key=lambda x: -x[1]):
        pattern_str = ", ".join(m.upper() for m in pattern) if pattern else "None (all present)"
        summary.append(f"  {count:3d} patients  —  Missing: {pattern_str}")
    summary += ["", "=" * 100, ""]

    lines = [
        "=" * 100, f"TOP-{top_k} NEIGHBOR REPORT  |  A_hat  |  first training batch",
        f"Batch size: {len(labels)} | Label distribution: {label_dist}", "=" * 100,
    ]
    lines += summary
    lines.append("PER-PATIENT TOP-K NEIGHBOR DETAILS")
    lines.append("=" * 100)

    for p in per_patient:
        missing_str = ", ".join(m.upper() for m in p['missing']) or "None"
        hom_str = f"{p['homophily']:.2f}" if p['homophily'] is not None else "N/A"
        lines.append("")
        lines.append(f"Patient ID={p['pid']} | Label={LABEL_NAMES[p['label']]} | Homophily={hom_str}")
        lines.append(f"  Missing modalities: {missing_str}")
        lines.append(f"  Top-{top_k} neighbors (by A_hat weight):")
        row = A_hat[p['i']].copy(); row[p['i']] = 0.0
        rank = 0
        for j in p['top_j']:
            if row[j] <= 0:
                continue
            rank += 1
            match = 'SAME' if labels[j] == labels[p['i']] else 'DIFF'
            lines.append(f"    #{rank:02d}: ID={pids[j]} | Label={LABEL_NAMES[labels[j]]} ({match}) | w={row[j]:.4f}")
        if rank == 0:
            lines.append("    No positive-weight neighbors found.")

    txt_path = os.path.join(out_dir, f"top{top_k}_neighbors_fold{fold_idx}_batch1.txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {txt_path}")

    csv_path = os.path.join(out_dir, f"top{top_k}_neighbors_fold{fold_idx}_batch1.csv")
    csv_fields = [
        'Patient_ID', 'Patient_Label', 'Patient_Missing_Mods', 'Homophily',
        'Neighbor_Rank', 'Neighbor_ID', 'Neighbor_Label', 'Match', 'Edge_Weight', 'Neighbor_Missing_Mods',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for p in per_patient:
            base = {
                'Patient_ID': p['pid'], 'Patient_Label': LABEL_NAMES[p['label']],
                'Patient_Missing_Mods': ', '.join(m.upper() for m in p['missing']) or 'None',
                'Homophily': f"{p['homophily']:.3f}" if p['homophily'] is not None else 'N/A',
            }
            row_weights = A_hat[p['i']].copy(); row_weights[p['i']] = 0.0
            valid_neighbors = [(rank + 1, j) for rank, j in enumerate(p['top_j']) if row_weights[j] > 0]
            empty_base = {k: '' for k in base}
            if not valid_neighbors:
                w.writerow({**base, 'Neighbor_Rank': '', 'Neighbor_ID': '',
                            'Neighbor_Label': '', 'Match': '', 'Edge_Weight': '', 'Neighbor_Missing_Mods': ''})
            else:
                for idx, (rank, j) in enumerate(valid_neighbors):
                    n_missing = [m for m in modality_order if not patient_presence[m][j]]
                    patient_part = base if idx == 0 else empty_base
                    w.writerow({**patient_part,
                        'Neighbor_Rank': rank, 'Neighbor_ID': pids[j],
                        'Neighbor_Label': LABEL_NAMES[labels[j]], 'Match': 'SAME' if labels[j] == p['label'] else 'DIFF',
                        'Edge_Weight': f"{row_weights[j]:.4f}", 'Neighbor_Missing_Mods': ', '.join(m.upper() for m in n_missing) or 'None',
                    })
    print(f"Saved: {csv_path}")

    save_html(per_patient, A_hat, labels, pids, patient_presence, modality_order, out_dir)
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == '__main__':
    main()
