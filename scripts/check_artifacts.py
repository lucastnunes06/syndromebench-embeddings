"""Verify that source-provided aggregate artifacts agree with the recorded run.

This validates saved CSV/JSON files only. It does NOT rerun the private dataset.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def verify(output_dir=OUT):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    required = {'cosine', 'euclidean'}
    assert required <= set(manifest['metrics']), 'Missing metric result in manifest'
    eda = pd.read_csv(output_dir / 'eda_syndrome_counts.csv')
    expected_count = int(eda['n_images'].sum())
    assert len(eda) >= 2 and expected_count > 0, 'Insufficient data for evaluation'
    for metric in required:
        fold = pd.read_csv(output_dir / f'fold_metrics_{metric}.csv')
        assert len(fold) >= 2, f'{metric}: missing cross-validation folds'
        recorded = manifest['metrics'][metric]['mean_metrics']
        for key in ('macro_f1', 'macro_auc', 'top1_accuracy', 'top5_accuracy'):
            assert np.isclose(fold[key].mean(), recorded[key], atol=1e-10), \
                f'{metric}: {key} differs between manifest and fold outputs'
        assert all(0 <= x <= 1 for x in fold['macro_f1'])
        assert all(0 <= x <= 1 for x in fold['macro_auc'])
        conf = pd.read_csv(output_dir / f'confusion_matrix_{metric}.csv', index_col=0)
        assert conf.shape == (len(eda), len(eda)), f'{metric}: confusion matrix shape'
        assert int(conf.to_numpy().sum()) == expected_count, f'{metric}: confusion matrix total'
    base = pd.read_csv(output_dir / 'baseline_comparison.csv')
    assert 'Nearest centroid (L2)' in set(base['model'])
    print(f'Aggregate checks passed: {len(eda)} classes, {expected_count} images.')


if __name__ == '__main__':
    verify()
