"""Human post-hoc morphology interpretation ONLY. Never imported by training.
These examples are illustrative, not labels, pure clusters, or calibrated taxonomy.
"""
from run_experiments import *
from render_results import plt,COLORS

def main():
    curves=json.loads((ROOT/'data/raw_curves.json').read_text());assign=pd.read_csv(ROOT/'cluster_assignments.csv').set_index('curve_id')
    choices=[('IFO-Bridge-like','C096','Early rise, broad top, slow decline; only ~101 h observed.'),
             ('IFO-Hill-like','C004','Rise then substantial fall and a flatter ending; phases are relative.'),
             ('IFO-Slope-like','C182','Fast initial loss followed by a much slower decline.'),
             ('IFO-Valley-like','C059','Initial loss, recovery, then decline; slight final upturn retained.')]
    rows=[]
    for norm in ['minmax','maxabs']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=norm=='minmax')
        for (name,cid,note),ax,color in zip(choices,axes.flat,['#6b9464','#99aa4e','#edb54b','#dac856']):
            c=next(c for c in curves if c['id']==cid);m=c['meta'];cluster=int(assign.loc[cid,'primary_cluster'])
            ax.plot(c['u'],values(c,norm),'.-',color=color,lw=2,ms=4)
            ax.set_title(f'{name} | {cid} | global cluster {cluster}\nObserved span: {m["duration_h"]:.1f} h; amplitude/max: {100*m["relative_y_range"]:.2f}%')
            ax.set_xlabel('Relative observed progress (0-1)');ax.set_ylabel('Output / max absolute output' if norm=='maxabs' else '(Output - min) / (max - min)');ax.grid(alpha=.18)
            if norm=='minmax':rows.append({'posthoc_shape':name,'curve_id':cid,'primary_cluster':cluster,'duration_h':m['duration_h'],'relative_y_range':m['relative_y_range'],'source_path':m['source_path'],'interpretation':note,'used_for_training_or_selection':False})
        fig.suptitle('Four target-like observed examples (post-hoc interpretation, NOT four learned classes)\nAll original points; no smoothing; no common 200 h boundary is imposed.',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(ROOT/f'figures/00_four_target_examples_{norm}.png');plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'target_shape_examples.csv',index=False)
    dump(ROOT/'experiments/posthoc_policy.json',{'training_completed_before_examples':True,
        'examples_are_manual_interpretations':True,'examples_are_not_labels':True,
        'not_used_for_parameter_or_k_selection':True,'shared_200h_boundary_verified':False})
    # Model and source hashes make later changes detectable.
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'experiments').glob('*.npz'))}
    dump(ROOT/'experiments/model_sha256.json',manifest)
    # Flag scale concerns without removing curves or changing their values.
    inv=pd.read_csv(ROOT/'data/curve_inventory.csv')
    flags=[]
    for _,r in inv.iterrows():
        reasons=[]
        if pd.notna(r.duration_h) and r.duration_h>1e5:reasons.append('recorded time span >100000 h; suspect digitization scale, not a verified lifetime')
        if r.y_max<0:reasons.append('all output values negative; axis calibration/quantity needs checking')
        if r.axis_status!='time_verified':reasons.append('not eligible for real-hour comparison without additional metadata')
        if r.relative_y_range<.02:reasons.append('MinMax amplifies a <2% relative variation; inspect MaxAbs view')
        if r.duplicate_x_points>0:reasons.append('vertical segments preserved; zero-duration jumps have zero L2 measure')
        if reasons:flags.append({'curve_id':r.curve_id,'figure':r.figure,'flags':'; '.join(reasons),'retained':True})
    pd.DataFrame(flags).to_csv(ROOT/'data/quality_flags.csv',index=False)
    print(rows,flush=True)

if __name__=='__main__':main()
