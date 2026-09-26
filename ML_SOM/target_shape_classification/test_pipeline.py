import unittest
from pathlib import Path

import numpy as np

import redigitize
import run
import som_evidence
import unsupervised_som
import classification_decision_view
import delivery_coverage


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.t=np.linspace(0,100,61)

    def model(self,y,t=None):
        t=self.t if t is None else t
        return run.morphology(dict(t=t,t_hour=t,y=y),run.DEFAULT)

    def test_four_shapes_and_backgrounds(self):
        t=self.t
        examples={
            "bridge":np.interp(t,[0,15,40,100],[.72,1,1,.78]),
            "hill":np.interp(t,[0,20,40,100],[.6,1,.52,.4]),
            "slope":.3+.7*np.exp(-t/15)-.001*t,
            "valley":np.interp(t,[0,15,40,70,100],[.9,.35,.75,.82,.65]),
        }
        for label,y in examples.items():
            with self.subTest(label=label):
                m=self.model(y)
                self.assertEqual(m["candidate_class"],label)
                self.assertTrue(m["stages_complete"])
        self.assertFalse(self.model(np.ones_like(t))["stages_complete"])
        self.assertFalse(self.model(1-.5*t/100)["stages_complete"])

    def test_hour_scaling_changes_rates_not_shape(self):
        y=.3+.7*np.exp(-self.t/15)-.001*self.t
        a=self.model(y)
        b=self.model(y,self.t*24)
        self.assertEqual(a["candidate_class"],b["candidate_class"])
        np.testing.assert_allclose(np.asarray(a["landmark_hours"])*24,b["landmark_hours"])
        np.testing.assert_allclose(np.asarray(a["stage_slopes_per_h"])/24,b["stage_slopes_per_h"])

    def test_one_percent_fluctuations_remain_in_observations_and_features(self):
        base=.3+.7*np.exp(-self.t/15)-.001*self.t
        y=base.copy()
        y[45]+= .012
        y[50]-= .012
        c=dict(t=self.t,t_hour=self.t,y=y)
        self.assertEqual(c["y"][45],y[45])
        m=run.morphology(c,run.DEFAULT)
        self.assertGreaterEqual(m["one_to_three_percent_extrema"],2)
        self.assertEqual(m["candidate_class"],"slope")

    def test_som_features_retain_source_point_one_percent_wiggle(self):
        grid=np.linspace(0,1,96)
        smooth=1-.2*grid
        phase=np.linspace(0,1,97)
        source=1-.2*phase
        wiggly=source.copy()
        wiggly[48]+= .015
        wiggly[49]-= .015
        matrix,_,_,events=unsupervised_som.features(
            np.vstack([smooth,smooth]),[(phase,source),(phase,wiggly)])
        self.assertEqual(events[0],0)
        self.assertGreaterEqual(events[1],2)
        self.assertGreater(np.linalg.norm(matrix[0]-matrix[1]),.01)
        early=source.copy()
        early[2]+=.015
        early_features,_,_,early_events=unsupervised_som.features(
            np.vstack([smooth,smooth]),[(phase,source),(phase,early)])
        self.assertGreaterEqual(early_events[1],1)
        self.assertGreater(np.linalg.norm(early_features[0]-early_features[1]),.01)

    def test_som_drawup_feature_keeps_source_recovery(self):
        grid=np.linspace(0,1,96)
        smooth=1-.3*grid
        phase=np.linspace(0,1,97)
        source=1-.3*phase
        recovered=source.copy()
        recovered[48]+=.012
        base,_,_,_=unsupervised_som.features(
            np.vstack([smooth,smooth]),[(phase,source),(phase,recovered)],
            drawup_weight=.5)
        self.assertGreater(np.linalg.norm(base[0]-base[1]),.01)
        np.testing.assert_allclose(base[0,-16:],0)
        self.assertGreater(np.max(base[1,-16:]),0)

    def test_som_seed_stability_does_not_replace_primary_candidate(self):
        rows={seed:[dict(file_id="F00001",som_candidate_class="hill" if seed==42 else "bridge")]
              for seed in (41,42,43,44,45)}
        result,_=unsupervised_som.consensus(rows,42)
        self.assertEqual(result[0]["som_candidate_class"],"hill")
        self.assertEqual(result[0]["ensemble_majority_class"],"bridge")
        self.assertEqual(result[0]["som_stability"],"variable")

    def test_seed_selection_balances_stage_motifs_within_topology_band(self):
        quality={41:{"topographic_error":.01,"quantization_error":.2},
                 42:{"topographic_error":.013,"quantization_error":.19},
                 43:{"topographic_error":.04,"quantization_error":.1}}
        scores={41:.6,42:.8,43:1.0}
        self.assertEqual(unsupervised_som.select_seed([41,42,43],quality,scores,.005),42)
        self.assertEqual(unsupervised_som.select_seed([41,42,43],quality,scores,0),41)

    def test_early_stage_sampling_balances_unlabeled_rare_shape(self):
        labels=['bridge']*12+['hill']+['']
        indices=np.arange(len(labels))
        factors=unsupervised_som.stage_sampling_factors(indices,1,labels,2)
        self.assertEqual(factors[0],1)
        self.assertEqual(factors[12],15)
        self.assertEqual(factors[13],1)
        np.testing.assert_array_equal(
            unsupervised_som.stage_sampling_factors(indices,0,labels,2),
            np.ones(len(labels)))

    def test_coupled_sampling_preserves_draws_when_one_curve_is_removed(self):
        branch=np.arange(4)
        full=unsupervised_som.sample_training_indices(
            branch,branch,np.ones(4)/4,42,1000,'gumbel')
        retained=np.array([0,1,3])
        reduced=unsupervised_som.sample_training_indices(
            retained,branch,np.ones(3)/3,42,1000,'gumbel')
        np.testing.assert_array_equal(full[full!=2],reduced[full!=2])

    def test_som_motif_retrieval_uses_neuron_distance(self):
        features=np.array([[0.,0.],[1.,0.]])
        branch=np.array([0,0])
        metadata=[dict(file_id="A",source_group="doi/a",source_csv="a.csv",
                       source_image="a.png",quality_flags="",analysis_version="a.csv"),
                  dict(file_id="B",source_group="doi/b",source_csv="b.csv",
                       source_image="b.png",quality_flags="",analysis_version="b.csv")]
        arrays={"seed_42_no_early_gain_codebook":features.copy(),
                "seed_42_early_gain_codebook":features.copy()}
        prototypes=[dict(seed=42,branch="no_early_gain",neuron=0,label="slope"),
                    dict(seed=42,branch="no_early_gain",neuron=1,label="valley")]
        rows,_=unsupervised_som.motif_retrieval(features,branch,metadata,[42],
                                                arrays,prototypes,[])
        ranks={(r["file_id"],r["target_class"]):r["rank_in_class"] for r in rows}
        self.assertEqual(ranks[("A","slope")],1)
        self.assertEqual(ranks[("B","valley")],1)

    def test_source_audit_rejects_only_the_recorded_candidate_class(self):
        finding={"rejected_candidate_class":"valley"}
        self.assertEqual(som_evidence.evidence_status("valley","other",{},finding),
                         "source_audit_rejects_candidate")
        self.assertEqual(som_evidence.evidence_status("slope","other",{},finding),
                         "unreviewed_candidate")

    def test_decision_view_preserves_review_and_provisional_boundaries(self):
        base={"som_candidate_class":"hill","source_verified":"",
              "stages_verified":"","reviewed_class":"",
              "som_evidence_status":"unreviewed_candidate",
              "time_start_h":"0","time_end_h":"100"}
        stage={"stage_candidate_class":"hill","stage_complete":"True"}
        self.assertEqual(classification_decision_view.decide(base,stage),
                         ("hill","provisional_som_stage_agreement"))
        self.assertEqual(classification_decision_view.decide(
            base|{"som_evidence_status":"source_audit_rejects_candidate"},stage),
            ("unresolved","source_audit_or_shape_ambiguous"))
        self.assertEqual(classification_decision_view.decide(
            base|{"source_audit_resolution":"needs_pixel_level_check"},stage),
            ("unresolved","source_audit_pending"))
        self.assertEqual(classification_decision_view.decide(
            base|{"source_verified":"yes","stages_verified":"yes",
                  "reviewed_class":"valley"},stage),
            ("valley","source_verified"))
        self.assertEqual(classification_decision_view.decide(base,
            {"stage_candidate_class":"bridge","stage_complete":"True"}),
            ("unresolved","som_stage_unresolved_or_disagree"))
        self.assertEqual(classification_decision_view.decide(base,
            {"stage_candidate_class":"bridge","stage_complete":"True"},2),
            ("bridge","provisional_ensemble_stage_agreement"))
        self.assertEqual(classification_decision_view.decide(
            base|{"time_start_h":""},stage),
            ("unresolved","actual_hour_axis_unresolved"))

    def test_source_axis_reviews_prevent_power_and_equivalent_time_mislabeling(self):
        root=Path(__file__).resolve().parent.parent/"data_final"
        power=list((root/"final_data/x_time_h/10.1002_cssc.201702265").rglob("*Series_2.csv"))[0]
        info=run.source_info(power,root)
        self.assertEqual(info["y_kind"],"non_pce_or_unresolved")
        self.assertEqual(info["y_name"],"MPP")
        equivalent=list((root/"final_data/x_time_h/10.1038_s41467-024-46145-7").rglob("*.csv"))[0]
        info=run.source_info(equivalent,root)
        self.assertIsNone(info["time_factor"])
        self.assertEqual(info["time_unit"],"equivalent_hours")
        mixed=root/"data_all/x_time_day/10.1002_adfm.201804128/Fig8_C_stability_performance_curve_decay/accepted"
        pce=list(mixed.glob("*__PCE__*.csv"))[0]
        ff=list(mixed.glob("*__FF__*.csv"))[0]
        self.assertEqual(run.source_info(pce,root)["y_kind"],"pce")
        self.assertNotEqual(run.source_info(ff,root)["y_kind"],"pce")
        eta=list((root/"data_all/x_time_h/10.1021_acsami.0c14218").rglob("*.csv"))[0]
        self.assertEqual(run.source_info(eta,root)["y_kind"],"pce")
        misleading=root/"final_data/x_time_h/10.1039_d5el00182j/Fig2_E_stability_performance_curve_voc/accepted"
        normalized_pce=list(misleading.glob("*.csv"))[0]
        info=run.source_info(normalized_pce,root)
        self.assertEqual((info["y_kind"],info["y_name"]),("pce","Norm. PCE"))
        eta_final=root/"final_data/x_time_h/10.1002_adfm.201602803/Fig4_B_jsc/accepted"
        info=run.source_info(list(eta_final.glob("*.csv"))[0],root)
        self.assertEqual((info["y_kind"],info["y_name"]),("pce","η"))

    def test_delivery_cap_keeps_disputed_and_non_hour_files_out_of_four_groups(self):
        base={"file_id":"F00001","decision_group":"unresolved",
              "decision_basis":"som_stage_unresolved_or_disagree",
              "som_candidate_class":"hill","time_start_h":"0","time_end_h":"100"}
        self.assertEqual(delivery_coverage.delivery_group(base),
                         ("hill","nearest_som_only_stage_unresolved","assigned"))
        self.assertEqual(delivery_coverage.delivery_group(
            base|{"decision_basis":"source_audit_pending"})[0],
            "extended_review_needed")
        self.assertEqual(delivery_coverage.delivery_group(
            base|{"time_end_h":""})[0],"excluded")

    def test_calibration_linear_and_log(self):
        linear={"pixel_1":10,"value_1":0,"pixel_2":110,"value_2":100,"scale":"linear"}
        log={"pixel_1":10,"value_1":1,"pixel_2":110,"value_2":100,"scale":"log10"}
        self.assertAlmostEqual(redigitize.axis_value(60,linear),50)
        self.assertAlmostEqual(redigitize.axis_value(60,log),10)
        with self.assertRaises(ValueError):
            redigitize.axis_value(60,linear|{"pixel_2":10})

    def test_time_unit_days_and_approximate_months_in_hours(self):
        self.assertEqual(run.axis_hours("data_all",{"name":"Time","unit":"d"})[0],24)
        factor, unit, note = run.axis_hours("data_all",{"name":"Time","unit":"months"})
        self.assertAlmostEqual(factor, 730.485)
        self.assertEqual(unit, "months_approx")
        self.assertIn("approximate", note)
        self.assertIsNone(run.axis_hours("data_all",{"name":"Time","unit":"cycles"})[0])
        self.assertEqual(run.y_kind({"name":"Normalized effeciency"}),"pce")
        self.assertEqual(run.y_kind({"name":"η (%)"}),"pce")
        self.assertEqual(run.y_kind({"name":"Δ PCE"}),"derived_delta_pce")

    def test_dtw_value_and_derivative(self):
        a=np.linspace(0,1,64)**2
        self.assertAlmostEqual(run.dtw_distance(a,a,.1,.5),0)
        self.assertGreater(run.dtw_distance(a,a[::-1],.1,.5),.1)


if __name__=="__main__":
    unittest.main()
