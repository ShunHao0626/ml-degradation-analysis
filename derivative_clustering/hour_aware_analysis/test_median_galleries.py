import numpy as np
import pandas as pd

from run_analysis import pointwise_hour_median


def curve(t,y):
    return pd.DataFrame({"time_h":t,"y_max_normalized":y})


def test_median_does_not_pad_short_curves_or_stretch_time():
    short=curve([0.,1.],[0.,10.])
    long=curve([0.,2.],[0.,2.])
    result=pointwise_hour_median([short,long])
    np.testing.assert_allclose(result.time_h,[0,1,2])
    np.testing.assert_allclose(result.median_response,[0,5.5,2])
    np.testing.assert_array_equal(result.n_contributing,[2,2,1])
    np.testing.assert_allclose(result.coverage_fraction,[1,1,.5])


def test_median_respects_different_start_hours_and_equal_curve_weights():
    a=curve([0.,1.,2.],[0.,0.,0.])
    b=curve([1.,3.],[1.,1.])
    c=curve([1.,1.2,1.5,1.8,2.],[8.,8.,8.,8.,8.])
    before=c.copy(deep=True)
    result=pointwise_hour_median([a,b,c]).set_index("time_h")
    assert result.loc[0,"n_contributing"]==1
    assert result.loc[1,"median_response"]==1
    assert result.loc[3,"median_response"]==1
    assert result.loc[3,"n_contributing"]==1
    pd.testing.assert_frame_equal(c,before)


def test_median_independent_of_curve_order():
    curves=[curve([0.,1.],[.5,.7]),curve([0.,2.],[1.,.2]),curve([0.,3.],[.7,.1])]
    pd.testing.assert_frame_equal(pointwise_hour_median(curves),pointwise_hour_median(curves[::-1]))


def test_shape_trend_levels_are_clock_dilation_invariant():
    from plot_shape_trends import relative_progress_levels
    t=np.linspace(0,10,31)
    a=curve(t,.5+.5*np.exp(-t/2))
    b=curve(t*100,a.y_max_normalized.to_numpy())
    before=a.copy(deep=True)
    u,levels=relative_progress_levels([a,b])
    np.testing.assert_allclose(u,np.linspace(0,1,48))
    np.testing.assert_allclose(levels[0],levels[1],atol=1e-12)
    pd.testing.assert_frame_equal(a,before)


def test_shape_trend_black_line_is_exact_cluster_median():
    from plot_shape_trends import cluster_medians
    levels=np.array([[1.,.8,.4],[.9,.5,.3],[.8,.7,.2],[.2,.4,.8]])
    labels=np.array([0,0,0,1])
    medians=cluster_medians(levels,labels)
    np.testing.assert_allclose(medians[0],[.9,.7,.3])
    np.testing.assert_allclose(medians[1],levels[3])
