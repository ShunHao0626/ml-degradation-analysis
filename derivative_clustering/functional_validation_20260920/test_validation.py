import numpy as np
import pandas as pd
import pytest

import validate as v


def test_compiled_dtw_matches_existing_implementations():
    rng=np.random.default_rng(44)
    for count in (1, 3):
        data=rng.normal(size=(4,13,count))
        weights=np.arange(count,0,-1,dtype=float)
        for radius in (0,3,13):
            fast=v.accelerated_dtw(data,weights,radius)
            cumulative=v.accelerated_dtw(data,weights,radius,False)
            for i in range(4):
                for j in range(4):
                    assert fast[i,j]==pytest.approx(v.prior.constrained_dtw(data[i],data[j],weights,radius),abs=1e-12)
                    assert cumulative[i,j]==pytest.approx(v.pfd_dtw.pfd_dtw_distance(data[i],data[j],weights=weights,window=radius),abs=1e-12)


def test_physical_time_scale_and_metadata_do_not_change_features():
    t=np.array([0.,.01,.05,.1,.3,.6,.8,1.])
    f=pd.DataFrame({"time_h":t,"y":1+.2*t-.1*t*t})
    modified=f.copy(); modified["time_h"]=f.time_h*3600+17
    modified["target_class"]="fabricated"; modified["is_injected_outlier"]=1
    a=v.representations([f])[0]; b=v.representations([modified])[0]
    np.testing.assert_allclose(a,b,atol=1e-9)


def test_linear_derivative_and_response_scaling():
    f=pd.DataFrame({"time_h":[0.,1.,4.,10.,30.,100.],"y":[1.,1.02,1.08,1.2,1.6,3.]})
    ch=v.representations([f])[0][0]
    np.testing.assert_allclose(ch[:,1],2/3,atol=1e-8)
    np.testing.assert_allclose(ch[:,2:],0,atol=1e-7)
    f["y"]*=10
    np.testing.assert_allclose(v.representations([f])[0][0],ch,atol=1e-8)


def test_aliases_duplicates_and_invalid_inputs():
    f=pd.DataFrame({"time_h":[0,1,2,3,4,4],"normalized_pce":[1,.9,.8,.7,.6,.6]})
    out=v.observed_frame(f)
    assert len(out)==5
    for broken in (f.iloc[:3], f.assign(normalized_pce=np.nan), f.assign(normalized_pce=0)):
        with pytest.raises(ValueError): v.observed_frame(broken)


def test_generators_are_reproducible_and_truncation_keeps_observations():
    a,truth,meta=v.new_batch("independent",31001,4)
    b,truth_b,_=v.new_batch("truncated",31001,4)
    c,truth_c,_=v.new_batch("independent",31001,4)
    np.testing.assert_array_equal(truth,truth_b)
    np.testing.assert_array_equal(truth,truth_c)
    for x,y,z in zip(a,b,c):
        pd.testing.assert_frame_equal(x,z)
        assert y.time_h.max()<=.55*x.time_h.max()
        assert len(y)>=5
        pd.testing.assert_frame_equal(x.loc[x.time_h<=.55*x.time_h.max()],y)


def test_background_contamination_counts_in_precision():
    truth=np.array(["bridge","hill","slope","valley","other_flat"])
    scores,_=v.assess(truth,np.array([0,1,2,3,0]),np.array(["short"]*5))
    assert scores["ari"]==1
    assert scores["macro_recall"]==1
    assert scores["macro_precision_with_background"]==pytest.approx(.875)


def test_unobserved_future_cannot_be_recovered_from_derivatives():
    # Exactly the same observed rise can later form a broad bridge or a hill.
    t=np.array([0,.03,.06,.09,.12,.18,.3,.6,1.])
    prefix=[.65,.75,.84,.93,1.]
    bridge=pd.DataFrame({"time_h":t,"y":prefix+[1.,.99,.96,.90]})
    hill=pd.DataFrame({"time_h":t,"y":prefix+[.85,.58,.5,.45]})
    observed=v.representations([bridge.iloc[:5],hill.iloc[:5]])[0]
    np.testing.assert_array_equal(observed[0],observed[1])
    full=v.representations([bridge,hill])[0]
    assert np.linalg.norm(full[0]-full[1])>0
