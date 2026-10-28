"""Build an offline-only dashboard from aggregate CSV/JSON metrics.

No network, raw embeddings, subject identifiers, or prediction serving.
"""
import json
from html import escape
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SyndromeBench | Evaluation Explorer</title>
<style>
:root{color-scheme:light;--ink:#14253a;--muted:#64778a;--line:#dce5ef;--nav:#0c2035;--blue:#315eaf;--teal:#087e80;--bg:#f4f7fb}
*{box-sizing:border-box}body{margin:0;font:15px/1.6 Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--nav);color:white;padding:24px max(26px,calc((100vw - 1150px)/2))}header small{color:#a6c1d9;letter-spacing:.16em;font-weight:800}header h1{font-size:34px;letter-spacing:-1.4px;line-height:1.1;margin:10px 0}header p{color:#cbd9e8;max-width:800px;margin:0}
main{max-width:1210px;margin:auto;padding:28px 24px 48px}.scope{font-size:13px;color:#495e72;background:#edf3fb;border-left:3px solid #7295c6;padding:12px 17px;margin-bottom:25px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:19px;box-shadow:0 5px 20px #243d5c05}.metric-label{font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.035em}.big{font-size:27px;letter-spacing:-1px;font-weight:780;margin-top:6px}.sub{font-size:12px;color:var(--muted)}.section{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;margin-top:18px}h2{font-size:18px;margin:0 0 11px}p{margin:8px 0 13px}.hint{font-size:12px;color:var(--muted)}.switch{display:flex;gap:8px;margin:12px 0 20px}.switch button{border-radius:30px;border:1px solid #b8c9dc;background:#fff;padding:8px 13px;cursor:pointer;color:var(--ink);font-weight:750}.switch button[aria-pressed="true"]{background:var(--nav);color:#fff;border-color:var(--nav)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #e8eef3;padding:9px 7px}th{color:var(--muted);font-size:11px;letter-spacing:.035em;text-transform:uppercase}.strong{font-weight:750}.plot{width:100%;border-radius:10px;display:block;object-fit:contain}.bars{display:grid;gap:13px}.barrow{display:grid;grid-template-columns:110px 1fr 52px;gap:9px;align-items:center;font-size:12px}.track{height:9px;border-radius:9px;background:#e8eef5;overflow:hidden}.fill{height:100%;background:var(--teal);border-radius:9px}a{color:var(--blue)}footer{font-size:12px;color:var(--muted);padding-top:30px}
@media(max-width:850px){.metrics{grid-template-columns:repeat(2,1fr)}.section{grid-template-columns:1fr}header h1{font-size:28px}}@media(max-width:470px){.metrics{grid-template-columns:repeat(2,1fr)}.card{padding:14px}.big{font-size:23px}main{padding:15px}}
</style></head><body>
<header><small>RESEARCH EVALUATION / STATIC EXPLORER</small><h1>SyndromeBench</h1><p>How do cosine and Euclidean KNN compare on image embeddings, and what do the baselines reveal?</p></header>
<main><div class="scope"><strong>Dataset notice.</strong> Figures were provided in an earlier research exercise. No individual predictions, medical diagnosis or live model calls. The original embeddings are not distributed, and these results have not been independently rerun in this build.</div>
<div class="metrics"><div class="card"><div class="metric-label">Image embeddings</div><div class="big">1,116</div><div class="sub">320 dimensions / embedding</div></div><div class="card"><div class="metric-label">Classes</div><div class="big">10</div><div class="sub">Syndrome identifiers</div></div><div class="card"><div class="metric-label">Evaluation</div><div class="big">10 folds</div><div class="sub">Grouped by individual</div></div><div class="card"><div class="metric-label">Best mean macro-F1</div><div class="big">0.768</div><div class="sub">Nearest-centroid baseline</div></div></div>
<div class="section"><section class="card"><h2>KNN: compare metrics</h2><p class="hint">Means across outer held-out folds. The technical report provides fold-level standard deviations.</p><div class="switch"><button id="cos" type="button" aria-pressed="true">Cosine + L2</button><button id="euc" type="button" aria-pressed="false">Euclidean + standard scaling</button></div><div id="metricbars" class="bars" aria-live="polite"></div><p id="compareNote" class="hint"></p></section>
<section class="card"><h2>Context: simple baselines</h2><p class="hint">Not every stronger result requires a more complex model.</p><table><thead><tr><th>Model</th><th>Macro-F1</th></tr></thead><tbody id="baselines"></tbody></table><p class="hint">This is not a clinical deployment benchmark.</p></section></div>
<div class="section"><section class="card"><h2>ROC curves across folds</h2><img class="plot" alt="Fold averaged one-vs-rest ROC comparing cosine and Euclidean distance" src="../outputs/roc_comparison.png"/><p class="hint">Macro one-vs-rest ROC; shaded bands represent fold variation.</p></section><section class="card"><h2>t-SNE projection</h2><img class="plot" alt="Two-dimensional t-SNE projection, points colored by syndrome ID" src="../outputs/tsne_by_syndrome.png"/><p class="hint">Exploratory visualization only; 2-D separation is not a diagnostic validation.</p></section></div>
<div class="section"><section class="card"><h2>Unequal class frequencies</h2><img class="plot" alt="Syndrome class image counts" src="../outputs/eda_images_per_syndrome.png"/><p class="hint">Unequal class counts motivate macro averaged metrics and per-class error analysis.</p></section><section class="card"><h2>Where errors occur</h2><img id="confusion" class="plot" src="../outputs/confusion_matrix_cosine.png" alt="Out of fold confusion matrix for the selected KNN"/><p class="hint">Row-normalized confusion shows missed cases for smaller classes. Toggle the KNN above to switch the matrix.</p></section></div>
<footer>Research demonstration only · Aggregated source-provided results · No clinical decisions · Review dataset permissions before distributing.</footer></main>
<script>
'use strict';
const metrics = DATA_INJECTED;
function clear(el){el.replaceChildren()}
function compare(metric){
 const record=metrics.knn[metric], bars=document.getElementById('metricbars');clear(bars);
 const specs=[['Macro-F1',record.f1],['Top-1 accuracy',record.top1],['Top-5 accuracy',record.top5],['Macro AUC',record.auc]];
 for(const [label,value] of specs){const row=document.createElement('div');row.className='barrow';
 const lab=document.createElement('span');lab.textContent=label;const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=`${Math.max(0,Math.min(100,value*100))}%`;track.append(fill);
 const num=document.createElement('strong');num.textContent=value.toFixed(3);row.append(lab,track,num);bars.append(row);}
 document.getElementById('cos').setAttribute('aria-pressed',metric==='cosine');
 document.getElementById('euc').setAttribute('aria-pressed',metric==='euclidean');
 document.getElementById('confusion').src=`../outputs/confusion_matrix_${metric}.png`;
 document.getElementById('compareNote').textContent=metric==='cosine'?'Cosine results use L2 row normalization. Some outer folds select k at the search ceiling of 15.':'Euclidean results use StandardScaler. This is a comparison of pipelines, not distances under identical preprocessing.';
}
const base=document.getElementById('baselines');
for(const item of metrics.baselines){const row=document.createElement('tr');const name=document.createElement('td');name.textContent=item.name;const val=document.createElement('td');val.className='strong';val.textContent=item.f1.toFixed(4);row.append(name,val);base.append(row)}
document.getElementById('cos').addEventListener('click',()=>compare('cosine'));
document.getElementById('euc').addEventListener('click',()=>compare('euclidean'));
compare('cosine');
</script></body></html>'''


def build(output_dir=OUT, target=None):
    out = Path(output_dir)
    summary = pd.read_csv(out/'performance_summary.csv')
    baselines = pd.read_csv(out/'baseline_comparison.csv')
    records = {}
    for record in summary.to_dict(orient='records'):
        label = record['Distance Metric'].lower()
        def first(x):
            return float(str(record[x]).split('±')[0].strip())
        records[label] = {'f1':first('Macro F1 (mean ± std)'),
                          'top1':first('Top-1 (mean ± std)'),
                          'top5':first('Top-5 (mean ± std)'),
                          'auc':first('Macro AUC (mean ± std)')}
    b = sorted([{'name':str(row['model']),'f1':float(row['macro_f1_mean'])} for row in baselines.to_dict(orient='records')],
               key=lambda x:-x['f1'])
    payload = json.dumps({'knn':records,'baselines':b},ensure_ascii=False)
    content = HTML.replace('DATA_INJECTED',payload)
    result=Path(target) if target else ROOT/'dashboard'/'index.html'
    result.parent.mkdir(parents=True,exist_ok=True)
    result.write_text(content,encoding='utf-8')
    print(f'Wrote {result} from aggregate data ({len(content)} characters).')
    return result


if __name__ == '__main__':
    build()
