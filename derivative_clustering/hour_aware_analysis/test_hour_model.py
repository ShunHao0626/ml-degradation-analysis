import numpy as np
import pandas as pd
import pytest

import hour_model as h


def test_native_hour_derivatives_known_quadratic():
    t=np.array([0.,.1,.4,1.,2.,3.,5.,8.,10.])
    y=1+.04*t+.003*t*t
    values=h.physical_local_polynomial(t,y)
    np.testing.assert_allclose(values[:,0],y,atol=1e-10)
    np.testing.assert_allclose(values[:,1],.04+.006*t,atol=1e-10)
    np.testing.assert_allclose(values[:,2],.006,atol=1e-10)


def test_clock_dilation_changes_hour_derivatives_and_response_times():
    t=np.r_[0.,np.geomspace(.01,20,100)]
    f=pd.DataFrame({"time_h":t,"y":.6+.4*np.exp(-t/2)})
    a,ca=h.physical_description(f)
    f.time_h*=100
    b,cb=h.physical_description(f)
    np.testing.assert_allclose(cb.dy_per_h,ca.dy_per_h/100,rtol=1e-7,atol=1e-11)
    np.testing.assert_allclose(cb.d2y_per_h2,ca.d2y_per_h2/10000,rtol=1e-7,atol=1e-11)
    assert a["excursion_10_status"]==b["excursion_10_status"]=="resolved"
    assert b["excursion_10_elapsed_h"]==pytest.approx(100*a["excursion_10_elapsed_h"])


def test_crossing_not_observed_is_censored_not_endpoint_filled():
    f=pd.DataFrame({"time_h":np.arange(8.),"y":1-.001*np.arange(8.)})
    a,_=h.physical_description(f)
    assert a["excursion_10_status"]=="not_observed_censored"
    assert np.isnan(a["excursion_10_elapsed_h"])
    assert a["excursion_10_censor_elapsed_h"]==7


def test_large_sampling_gap_remains_uncertain():
    event=h.first_excursion(np.array([0.,1.,100.,101.,102.]),np.array([1.,.98,.6,.59,.58]),.1)
    assert event["status"]=="interval_uncertain"
    assert event["bracket_start_h"]==1
    assert event["bracket_end_h"]==100


def test_longer_observation_does_not_rescale_early_response():
    t=np.arange(0.,10.01,.1)
    f=pd.DataFrame({"time_h":t,"y":.6+.4*np.exp(-t/2)})
    a,ca=h.physical_description(f)
    longer=np.r_[t,np.arange(11.,101.)]
    b,cb=h.physical_description(pd.DataFrame({"time_h":longer,"y":.6+.4*np.exp(-longer/2)}))
    assert b["observed_duration_h"]==10*a["observed_duration_h"]
    assert b["excursion_10_elapsed_h"]==pytest.approx(a["excursion_10_elapsed_h"],abs=1e-10)
    np.testing.assert_allclose(ca.dy_per_h.iloc[:10],cb.dy_per_h.iloc[:10],atol=1e-10)


def test_elapsed_landmarks_invariant_to_clock_origin():
    f=pd.DataFrame({"time_h":np.linspace(0,10,50),"y":1-.04*np.linspace(0,10,50)})
    a,_=h.physical_description(f); f.time_h+=50; b,_=h.physical_description(f)
    assert b["excursion_10_time_h"]==pytest.approx(a["excursion_10_time_h"]+50)
    assert b["excursion_10_elapsed_h"]==pytest.approx(a["excursion_10_elapsed_h"])


def test_metadata_and_raw_inputs_not_modified():
    t=np.linspace(0,10,30)
    f=pd.DataFrame({"time_h":t,"y":np.exp(-t/5),"target":"valley"})
    before=f.copy(deep=True)
    a,ca=h.physical_description(f)
    f.target="hill"; b,cb=h.physical_description(f)
    pd.testing.assert_frame_equal(ca,cb)
    np.testing.assert_array_equal(f.time_h,before.time_h)
    np.testing.assert_array_equal(f.y,before.y)


def test_subgroups_use_crossing_hour_not_total_duration():
    rng=np.random.default_rng(321)
    response=np.r_[rng.uniform(.8,1.2,12),rng.uniform(8,12,12),rng.uniform(80,120,12)]
    f=pd.DataFrame({"excursion_10_elapsed_h":response,"excursion_10_status":"resolved","observed_duration_h":rng.uniform(1000,2000,36)})
    labels,_,models=h.kinetic_subgroups(np.zeros(36,int),f)
    np.testing.assert_array_equal(labels,np.repeat([0,1,2],12))
    f.observed_duration_h*=100
    labels2,_,_=h.kinetic_subgroups(np.zeros(36,int),f)
    np.testing.assert_array_equal(labels,labels2)
    assert models["0"]["selected_k"]==3


def test_disconnected_graph_partition_and_roundoff_stability():
    rng=np.random.default_rng(1)
    features=np.repeat(np.arange(4)*100.,15)+rng.uniform(0,.1,60)
    distance=np.abs(features[:,None]-features[None,:])
    labels,info=h.stable_shape_partition(distance)
    assert info["connected_components"]==4
    assert info["eigen_residual_max"]<1e-10
    assert h.baseline.adjusted_rand_score(np.repeat(np.arange(4),15),labels)==1
    rounded,_=h.stable_shape_partition(np.round(distance,10))
    assert h.baseline.adjusted_rand_score(labels,rounded)==1


def test_identical_prefix_cannot_reveal_future_shape():
    t=np.arange(10.)
    # One possible future would recover, another would remain flat. Neither
    # future is provided; descriptors must remain identical and unresolved.
    f=pd.DataFrame({"time_h":t,"y":1-.002*t,"future_shape":"valley"})
    a,ca=h.physical_description(f)
    f.future_shape="slope"
    b,cb=h.physical_description(f)
    pd.testing.assert_frame_equal(ca,cb)
    assert a["excursion_10_status"]==b["excursion_10_status"]=="not_observed_censored"
