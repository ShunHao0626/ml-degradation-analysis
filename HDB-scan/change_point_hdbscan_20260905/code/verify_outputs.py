"""Independent invariants for source preservation, leakage, transformations and results."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/method1_mpl'
from core import *
import pickle, ast

ROOT=Path(__file__).resolve().parents[1]

def verify():
    checks=[]
    def check(name,condition,detail=''):
        if not bool(condition):raise AssertionError(f'{name}: {detail}')
        checks.append(dict(check=name,status='passed',detail=detail))
    qc=pd.read_csv(ROOT/'qc_summary.csv')
    check('all input curves audited',len(qc)==218)
    for _,r in qc.iterrows():
        p=Path(r.source_path)
        check(f'raw hash preserved {r.sample_id}',hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256)
        copied=ROOT/'raw_input'/p.relative_to(ROOT.parent/'samples_test')
        check(f'raw snapshot identical {r.sample_id}',copied.read_bytes()==p.read_bytes())
    model=pickle.loads((ROOT/'final'/'fitted_model.pkl').read_bytes())
    frame=pd.read_csv(ROOT/'final'/'feature_table.csv',index_col=0)
    assignment=pd.read_csv(ROOT/'final'/'cluster_assignment.csv');x=pd.read_csv(ROOT/'final'/'clustering_matrix.csv',index_col=0).to_numpy()
    check('sample alignment',assignment.sample_id.tolist()==frame.index.tolist()==model['ids'])
    check('finite clustering matrix',np.isfinite(x).all())
    check('labels agree with saved model',np.array_equal(assignment.cluster.to_numpy(),model['model'].labels_))
    check('membership in unit interval',assignment.membership_probability.between(0,1).all())
    check('no identifiers in features',all(not any(token in c for token in ['sample_id','figure','source','label','cluster']) for c in frame.columns))
    # Reconstruct saved preprocessing independently, including imputation and PCA.
    prep=model['preprocessing'];a=frame[prep['feature_names']].to_numpy();a=np.where(np.isfinite(a),a,prep['imputation_medians'])
    a=prep['scaler'].transform(a)
    if prep['pca'] is not None:a=np.einsum('ij,kj->ik',a-prep['pca'].mean_,prep['pca'].components_)
    check('saved preprocessing reproduces clustering coordinates',np.allclose(a,x,rtol=1e-9,atol=1e-9),f'max_abs_error={np.max(np.abs(a-x)):.3g}')
    cfg=json.loads((ROOT/'final'/'frozen_model.json').read_text())
    check('known-hour cohort only',assignment.unit.isin(['hours','days']).all())
    stage=cfg['candidate']['dataset'];w=cfg['selected_window_hours']
    for sid in frame.index:
        raw=pd.read_csv(ROOT/'cleaned_input'/f'{sid}.csv');window=pd.read_csv(ROOT/'window_inputs'/stage/f'{sid}.csv')
        expected=raw if w is None else raw[raw.elapsed<=w]
        check(f'unsmoothed window {sid}',np.allclose(expected.elapsed,window.elapsed) and np.allclose(expected.y,window.y))
        check(f'minimum raw points {sid}',len(window)>=12)
        if w is not None:check(f'covers entire candidate window {sid}',raw.elapsed.max()>=w)
    cp=pd.read_csv(ROOT/'stability'/'change_point_50x2_runs.csv')
    counts=cp.groupby(['sample_id','drop_fraction']).size()
    check('50 changepoint repeats at each drop fraction',len(counts)==2*len(frame) and counts.eq(50).all())
    for filename in ['sample_bootstrap_50_runs.csv','figure_bootstrap_50_runs.csv','repeated_seed_50_runs.csv']:
        d=pd.read_csv(ROOT/'stability'/filename);check(filename,len(d)==50)
    drops=pd.read_csv(ROOT/'stability'/'timepoint_deletion_100_runs.csv')
    check('100 final timepoint perturbation runs',len(drops)==100 and drops.groupby('drop_fraction').size().eq(50).all())
    cpgrid=pd.read_csv(ROOT/'search'/'cp_parameter_search.csv');check('252 CP configurations',len(cpgrid)==252)
    grid=pd.read_csv(ROOT/'search'/'hdbscan_parameter_search.csv')
    check('three normalization alternatives',set(grid.normalization)=={'initial','max','robust'})
    check('four feature representations',set(grid.representation.astype(str))=={'full','0.9','0.95','0.99'})
    check('both clustering metrics',set(grid.metric)=={'euclidean','manhattan'})
    check('six time-window strategies',grid.dataset.nunique()==6)
    audit=json.loads((ROOT/'unsupervised_audit_log.json').read_text())
    check('no smoothing or labels declared',not audit['smoothing'] and not audit['human_class_labels_used'] and audit['target_cluster_count'] is None)
    # Exact theoretical piecewise affine data exercises the actual-time linear design.
    t=np.r_[np.linspace(0,.35,25),np.linspace(.4,1,25)]
    y=np.where(t<.4,1+3*t,2.2-.7*(t-.4))
    testcfg=dict(id='TEST',algorithm='pelt',cost='linear',min_fraction=.05,penalty=.5)
    b=cp_detect(t,y,testcfg)
    check('actual-time linear changepoint detects slope change',np.min(np.abs(cp_times(t,b)-.375))<.06,str(b))
    check('affine normalization preserves calibrated changepoints',b==cp_detect(t,5*y+8,testcfg))
    check('constant raw curve yields zero changepoints',cp_detect(t,np.ones_like(t),testcfg)==[len(t)])
    # Check every linear interpolation against saved unsmoothed source observations.
    for f in (ROOT/'resampled_input').glob('S*.csv'):
        a=pd.read_csv(f);raw=pd.read_csv(ROOT/'window_inputs'/stage/f.name)
        check(f'linear grid no extrapolation {f.stem}',a.elapsed.min()>=raw.elapsed.min()-1e-9 and a.elapsed.max()<=raw.elapsed.max()+1e-9)
        check(f'linear interpolation reproduces saved data {f.stem}',np.allclose(a.y,np.interp(a.elapsed,raw.elapsed,raw.y)))
    # Verify the typical sampling resolution constraint independently.
    gi=json.loads((ROOT/'resampled_input'/'grid_info.json').read_text());dts=[]
    for f in (ROOT/'window_inputs'/stage).glob('*.csv'):
        a=pd.read_csv(f);tt=a.elapsed.to_numpy();dts.append(np.median(np.diff(tt/(tt[-1] if w is None else 1))))
    check('resampling grid no finer than typical native spacing',gi['grid_spacing']+1e-12>=np.median(dts))
    reps=pd.read_csv(ROOT/'final'/'representatives.csv')
    check('representatives are real dataset curves',set(reps.sample_id)<=set(frame.index))
    check('one medoid per reported group',reps.groupby('cluster').is_medoid.sum().eq(1).all())
    for sid in frame.index:
        seg=pd.read_csv(ROOT/'final'/'all_segment_descriptors.csv');ss=seg[seg.sample_id==sid]
        check(f'segments preserve raw point counts {sid}',ss.raw_point_count.sum()==len(pd.read_csv(ROOT/'window_inputs'/stage/f'{sid}.csv')))
    dump(dict(n_checks=len(checks),all_passed=True,checks=checks),ROOT/'verification_results.json')
    manifest=[]
    for f in sorted(ROOT.rglob('*')):
        if f.is_file() and not any(part in ['.venv','cache','__pycache__'] for part in f.relative_to(ROOT).parts) and f.name!='checksums.sha256' and f.suffix!='.log':
            manifest.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(ROOT)))
    (ROOT/'checksums.sha256').write_text('\n'.join(manifest)+'\n')
    print(f'PASS: {len(checks)} verification checks; all raw files preserved and outputs reproducible.')

if __name__=='__main__':verify()
