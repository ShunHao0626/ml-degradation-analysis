"""Verify newly exported presentation data without rerunning model selection."""
from pathlib import Path
import json,hashlib
from html.parser import HTMLParser
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result'

def main():
    summary=json.loads((OUT/'hdbscan_summary.json').read_text());a=pd.read_csv(ROOT/'final'/'cluster_assignment.csv');points=pd.read_csv(OUT/'plotted_curve_points.csv');means=pd.read_csv(OUT/'cluster_mean_curves_display_only.csv')
    frozen=json.loads((ROOT/'final'/'frozen_model.json').read_text());dataset=frozen['candidate']['dataset'];norm=frozen['candidate']['normalization']
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    check('frozen assignments unchanged',hashlib.sha256((ROOT/'final'/'cluster_assignment.csv').read_bytes()).hexdigest()==summary['source_assignment_sha256'])
    check('all 79 real samples included',set(points.sample_id)==set(a.sample_id))
    check('actual groups preserved',points.groupby('sample_id').cluster.first().sort_index().equals(a.set_index('sample_id').cluster.sort_index()))
    for sid in a.sample_id:
        p=points[points.sample_id==sid];r=pd.read_csv(ROOT/'window_inputs'/dataset/f'{sid}.csv');m=pd.read_csv(ROOT/'normalized_input'/dataset/norm/f'{sid}.csv')
        check(f'{sid}: raw observations unchanged',len(p)==len(r) and np.allclose(p.time_hours,r.elapsed) and np.allclose(p.raw_output,r.y))
        check(f'{sid}: display normalization correct',np.allclose(p.relative_to_initial,r.y/r.y.iloc[0]))
        check(f'{sid}: model normalization unchanged',np.allclose(p.model_normalized,m.normalized_y))
    for (label,mode),rows in means.groupby(['group','display_normalization']):
        ids=a.loc[a.cluster==label,'sample_id'];grid=rows.time_hours.to_numpy();aligned=[]
        for sid in ids:
            p=points[points.sample_id==sid];col='relative_to_initial' if mode=='initial' else 'model_normalized'
            check(f'{label}/{mode}/{sid}: no mean extrapolation',grid.min()>=p.time_hours.min()-1e-9 and grid.max()<=p.time_hours.max()+1e-9)
            aligned.append(np.interp(grid,p.time_hours,p[col]))
        check(f'{label}/{mode}: correct post-fit ensemble mean',np.allclose(np.mean(aligned,axis=0),rows.ensemble_mean))
        check(f'{label}/{mode}: constant contributing cohort',rows.n_contributors.eq(len(ids)).all())
    dt=np.median([np.median(np.diff(p.time_hours)) for _,p in points.groupby('sample_id')])
    check('mean grid respects native resolution',summary['mean_alignment']['step_hours']>=dt-1e-9)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            self.targets.extend(value for key,value in attrs if key in ['href','src'] and value)
    for file in OUT.glob('*.html'):
        parser=Links();parser.feed(file.read_text())
        for target in parser.targets:
            if target.startswith(('#','data:','http:','https:','javascript:')):continue
            check(f'{file.name}: local resource {target}',(file.parent/target).exists())
    for label,n in summary['cluster_counts'].items():
        stem='hdbscan_noise' if int(label)<0 else f'hdbscan_cluster_{label}'
        s=(OUT/(stem+'_interactive.html')).read_text();pos=s.index('Plotly.newPlot(');start=s.index('[',pos)
        traces,_=json.JSONDecoder().raw_decode(s[start:]);check(f'{stem}: expected interactive trace count',len(traces)==n+2)
        for trace in traces[:n]:
            p=points[points.sample_id==trace['name']]
            check(f'{stem}/{trace["name"]}: interactive original points',np.allclose(trace['x'],p.time_hours) and np.allclose(trace['y'],p.relative_to_initial))
    result=dict(all_passed=True,n_checks=len(checks),checks=checks,browser_interaction_test='Not run: browser runtime returned no available browsers',
        verification_scope='PNG visual review; HTML resource/trace validation; frozen-label and plotted-point/ensemble-mean identity')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=[]
    for file in sorted(ROOT.rglob('*')):
        if file.is_file() and not any(p in ['.venv','cache','__pycache__'] for p in file.relative_to(ROOT).parts) and file.name!='checksums.sha256' and file.suffix!='.log':
            lines.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(ROOT)))
    (ROOT/'checksums.sha256').write_text('\n'.join(lines)+'\n')
    print(f'PASS: {len(checks)} gallery checks. Frozen labels and original curve points preserved; means and offline HTML data verified.')

if __name__=='__main__':main()
