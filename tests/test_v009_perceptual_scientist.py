from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import numpy as np

from arc3lab.memory.affordance import AffordanceMemory
from arc3lab.memory.visual_belief import VisualBeliefState
from arc3lab.model.adapter import OpenAICompatLocalAdapter
from arc3lab.perception.scene import build_scene
from arc3lab.perception.spatial import SpatialControlModel, actor_key
from arc3lab.perception.state_estimator import PerceptualStateEstimator, _articulation_points
from arc3lab.perception.visual import VisualTracker
from arc3lab.policy.perceptual_coding import PerceptualDecisionPolicy
from arc3lab.types import ActionSpec


class Frame:
    def __init__(self, grid, actions=(1,2,3,4,6), levels=0):
        self.frame=[np.asarray(grid,dtype=np.int8)]
        self.available_actions=actions
        self.levels_completed=levels
        self.state='NOT_FINISHED'


def scene(grid, actions=(1,2,3,4,6)):
    return build_scene(Frame(grid, actions))


def move(grid, src, dst):
    out=np.asarray(grid,dtype=np.int8).copy(); color=int(out[src]); out[src]=0; out[dst]=color; return out


def seeded_world():
    g=np.zeros((10,10),dtype=np.int8); g[5,2]=7; g[5,8]=3
    control=SpatialControlModel(); cur=g.copy()
    seq=[(1,(5,2),(4,2)),(2,(4,2),(4,3)),(3,(4,3),(5,3)),(4,(5,3),(5,2))]
    for aid,src,dst in seq:
        nxt=move(cur,src,dst); control.observe(scene(cur,(1,2,3,4)), scene(nxt,(1,2,3,4)), ActionSpec(aid)); cur=nxt
    return g,control


def test_symmetry_and_goal_uncertainty():
    g=np.zeros((9,9),dtype=np.int8); g[4,2]=7; g[4,6]=7
    s=scene(g,(1,2))
    tracker=VisualTracker(); tracker.observe(s,step=0)
    est=PerceptualStateEstimator()
    summary=est.summarize(s,tracker,SpatialControlModel(),AffordanceMemory(),VisualBeliefState(),recent_grids=[g])
    assert summary['symmetry']['vertical'] == 1.0
    assert summary['uncertainty']['goal'] == 1.0


def test_cycle_detection():
    est=PerceptualStateEstimator()
    tracker=VisualTracker(); ctrl=SpatialControlModel(); aff=AffordanceMemory(); beliefs=VisualBeliefState()
    a=np.zeros((6,6),dtype=np.int8); a[2,2]=2
    b=np.zeros((6,6),dtype=np.int8); b[2,3]=2
    for i,g in enumerate([a,b,a,b]):
        s=scene(g,(1,2)); tracker.observe(s,step=i); out=est.summarize(s,tracker,ctrl,aff,beliefs,recent_grids=[g])
    assert out['temporal_phase']['period'] == 2
    assert out['recommended_mode'] in {'IDENTIFY_AGENCY','IDENTIFY_GOAL','DISCRIMINATE_DYNAMICS','REASON_ABOUT_PHASE'}


def test_role_hypotheses_find_actor_and_goal():
    g,ctrl=seeded_world(); s=scene(g,(1,2,3,4)); tracker=VisualTracker(); tracker.observe(s,step=0)
    beliefs=VisualBeliefState(); target=next(i for i,c in enumerate(s.components) if c.color==3)
    beliefs.update_from_model([{'target_object':target,'relation':'touch','confidence':.95,'evidence':'test'}],s,tracker,level=0)
    summary=PerceptualStateEstimator().summarize(s,tracker,ctrl,AffordanceMemory(),beliefs,recent_grids=[g])
    roles=summary['role_hypotheses']
    assert any(r['best_role']=='controlled' and r['role_scores']['controlled']>.7 for r in roles)
    assert any(r['role_scores']['goal_target']>.9 for r in roles)


def test_action_entropy_declines_with_consistency():
    g=np.zeros((6,6),dtype=np.int8); g[2,2]=2; s0=scene(g,(1,))
    aff=AffordanceMemory(); s1=scene(move(g,(2,2),(2,3)),(1,))
    for _ in range(5): aff.observe(s0,s1,ActionSpec(1))
    tracker=VisualTracker(); tracker.observe(s0,step=0)
    out=PerceptualStateEstimator().summarize(s0,tracker,SpatialControlModel(),aff,VisualBeliefState(),recent_grids=[g])
    assert out['action_models']['1']['effect_entropy'] == 0.0
    assert out['uncertainty']['action_model'] == 0.0


def test_topology_articulation_points_large_grid_no_recursion():
    mask=np.ones((64,64),dtype=bool)
    allowed={(r,c) for r in range(64) for c in range(64)}
    arts=_articulation_points(mask,allowed)
    assert arts == set()


def test_bottleneck_detected():
    mask=np.zeros((8,8),dtype=bool)
    mask[1:7,1:3]=True; mask[1:7,5:7]=True; mask[4,3:6]=True
    allowed={(r,c) for r,c in zip(*np.nonzero(mask))}
    arts=_articulation_points(mask,allowed)
    assert (4,4) in arts


def test_multiview_adapter_emits_two_images_and_labels():
    adapter=OpenAICompatLocalAdapter(image_side=256)
    current=np.zeros((8,8),dtype=np.int8); current[2,2]=3
    temporal=np.zeros((17,17),dtype=np.int8); temporal[4,4]=7
    class Resp:
        ok=True
        def raise_for_status(self): pass
        def json(self): return {'choices':[{'message':{'content':'{}'}}]}
    with patch('requests.post', return_value=Resp()) as post:
        adapter.complete('sys','user',grid={'views':[
            {'label':'CURRENT HIGH-RESOLUTION BOARD','grid':current,'side':512},
            {'label':'TEMPORAL CONTEXT','grid':temporal,'side':384},
        ]})
    payload=post.call_args.kwargs['json']
    content=payload['messages'][1]['content']
    images=[x for x in content if x['type']=='image_url']
    texts=[x['text'] for x in content if x['type']=='text']
    assert len(images)==2
    assert 'CURRENT HIGH-RESOLUTION BOARD' in texts
    assert 'TEMPORAL CONTEXT' in texts
    assert all(x['image_url']['detail']=='high' for x in images)


def test_perceptual_policy_uses_multiview_configuration():
    p=PerceptualDecisionPolicy(model=None,current_view_side=512,temporal_view_side=384)
    assert p.current_view_side==512 and p.temporal_view_side==384
    assert p.multiview_calls==0


def test_open_grid_topology_summary():
    g,ctrl=seeded_world(); s=scene(g,(1,2,3,4)); tracker=VisualTracker(); tracker.observe(s,step=0)
    out=PerceptualStateEstimator().summarize(s,tracker,ctrl,AffordanceMemory(),VisualBeliefState(),recent_grids=[g])
    assert out['topology']['available']
    assert out['topology']['reachable_ratio'] > .9
    assert out['topology']['articulation_points'] == 0


def test_new_track_novelty_and_change_bbox():
    g0=np.zeros((8,8),dtype=np.int8); g0[2,2]=2
    g1=g0.copy(); g1[6,6]=4
    tracker=VisualTracker(); tracker.observe(scene(g0),step=0); tracker.observe(scene(g1),step=1)
    out=PerceptualStateEstimator().summarize(scene(g1),tracker,SpatialControlModel(),AffordanceMemory(),VisualBeliefState(),recent_grids=[g0,g1])
    assert out['novelty']['new_tracks_now'] >= 1
    assert out['change']['changed_bbox'] == [6,6,6,6]


def test_topology_delta_records_gate_opening_action():
    g,ctrl=seeded_world()
    g[:,5]=8; g[5,5]=8
    s0=scene(g,(1,2,3,4)); tr=VisualTracker(); tr.observe(s0,step=0)
    est=PerceptualStateEstimator(); est.summarize(s0,tr,ctrl,AffordanceMemory(),VisualBeliefState(),recent_grids=[g])
    opened=g.copy(); opened[5,5]=0
    s1=scene(opened,(1,2,3,4)); tr.observe(s1,step=1,action=ActionSpec(2))
    out=est.summarize(s1,tr,ctrl,AffordanceMemory(),VisualBeliefState(),recent_grids=[g,opened],last_action=ActionSpec(2))
    delta=out['topology']['delta_from_previous']
    assert delta is not None
    assert delta['reachable_anchors'] >= 0
    assert out['topology']['recent_topology_events']
    assert out['topology']['recent_topology_events'][-1]['action'] == 2


def test_symmetry_anomaly_localizes_asymmetric_object():
    g=np.zeros((9,9),dtype=np.int8)
    g[4,2]=2; g[4,6]=2
    g[1,1]=3; g[1,7]=3
    g[7,4]=5
    g[2,3]=7
    s=scene(g,(1,2))
    tr=VisualTracker(); tr.observe(s,step=0)
    out=PerceptualStateEstimator().summarize(s,tr,SpatialControlModel(),AffordanceMemory(),VisualBeliefState(),recent_grids=[g])
    anomaly=out['symmetry']['anomaly']
    assert anomaly is not None
    assert anomaly['mismatch_cells'] > 0
    assert anomaly['objects_intersecting_mismatch']


def test_causal_interaction_graph_summarizes_action_tracks():
    g0=np.zeros((8,8),dtype=np.int8); g0[4,2]=7
    g1=move(g0,(4,2),(4,3))
    tr=VisualTracker(); tr.observe(scene(g0),step=0); tr.observe(scene(g1),step=1,action=ActionSpec(2))
    out=PerceptualStateEstimator().summarize(scene(g1),tr,SpatialControlModel(),AffordanceMemory(),VisualBeliefState(),recent_grids=[g0,g1],last_action=ActionSpec(2))
    assert '2' in out['causal_interaction_graph']
    assert out['causal_interaction_graph']['2']['event_counts']['move'] >= 1


def test_candidate_scores_are_removed_from_v009_context():
    g,ctrl=seeded_world(); s=scene(g,(1,2,3,4))
    p=PerceptualDecisionPolicy(model=None)
    p.spatial_control=ctrl
    p.visual_tracker.observe(s,step=0)
    from arc3lab.planning.counterfactual import enumerate_decision_candidates
    c=enumerate_decision_candidates(s,p.spatial_control,p.visual_tracker,p.affordances,p.visual_beliefs)
    ctx=p._visual_context(s,c)
    assert ctx['decision_candidates']
    assert all('score' not in row for row in ctx['decision_candidates'])


def test_high_orientation_entropy_forces_reasoning_when_model_exists():
    class M:
        def complete(self,*a,**k): return '{}'
    p=PerceptualDecisionPolicy(model=M(),max_model_calls=5)
    g=np.zeros((8,8),dtype=np.int8); g[2,2]=2
    s=scene(g,(1,2))
    p.last_perceptual_state={'orientation_entropy':.9,'recommended_mode':'IDENTIFY_AGENCY'}
    assert p._should_reason(s)


def test_reset_clears_temporal_estimator_not_affordances():
    p=PerceptualDecisionPolicy(model=None)
    p.perceptual_estimator.signature_history.append('x')
    p.affordances.action_effects[1]['change']=3
    p.on_level_reset()
    assert list(p.perceptual_estimator.signature_history)==[]
    assert p.affordances.action_effects[1]['change']==3


def test_frontier_graph_prefers_untested_local_candidate_and_records_edge():
    from arc3lab.planning.counterfactual import enumerate_decision_candidates
    from arc3lab.planning.frontier import ExplorationFrontier
    g=np.zeros((7,7),dtype=np.int8); g[3,3]=7
    s0=scene(g,(1,2))
    tr=VisualTracker(); tr.observe(s0,step=0)
    frontier=ExplorationFrontier()
    c0=enumerate_decision_candidates(s0,SpatialControlModel(),tr,AffordanceMemory(),VisualBeliefState())
    candidate,why=frontier.fallback_candidate(s0,c0)
    assert candidate is not None and candidate.spec is not None
    assert why == 'local_untested_frontier'
    key=frontier.candidate_key(s0,candidate)
    g1=move(g,(3,3),(3,4))
    s1=scene(g1,(1,2))
    frontier.observe_transition(s0,candidate.spec,s1,effect='change')
    assert key in frontier.nodes[frontier.state_key(s0)].tested_keys


def test_frontier_graph_finds_safe_route_to_remote_frontier():
    from arc3lab.planning.counterfactual import enumerate_decision_candidates
    from arc3lab.planning.frontier import ExplorationFrontier
    tr=VisualTracker(); aff=AffordanceMemory(); beliefs=VisualBeliefState(); ctrl=SpatialControlModel()
    g0=np.zeros((7,7),dtype=np.int8); g0[3,2]=7
    g1=move(g0,(3,2),(3,3))
    s0,s1=scene(g0,(1,)),scene(g1,(1,2))
    frontier=ExplorationFrontier()
    c0=enumerate_decision_candidates(s0,ctrl,tr,aff,beliefs)
    frontier.observe_state(s0,c0)
    spec0=next(c.spec for c in c0 if c.spec is not None and c.spec.action_id==1)
    frontier.observe_transition(s0,spec0,s1,effect='change')
    c1=enumerate_decision_candidates(s1,ctrl,tr,aff,beliefs)
    frontier.observe_state(s1,c1)
    summary=frontier.summary(s0,c0)
    assert summary['nearest_frontier_distance'] == 1
    assert summary['safe_route_first_candidate'] is not None


def test_frontier_routes_are_level_local():
    from arc3lab.planning.frontier import ExplorationFrontier
    g=np.zeros((5,5),dtype=np.int8); g[2,2]=7
    s0=build_scene(Frame(g,(1,),levels=0))
    s1=build_scene(Frame(g,(1,),levels=1))
    frontier=ExplorationFrontier()
    assert frontier.state_key(s0) != frontier.state_key(s1)


def test_perceptual_context_exposes_exploration_graph_without_score_policy():
    from arc3lab.planning.counterfactual import enumerate_decision_candidates
    g,ctrl=seeded_world(); s=scene(g,(1,2,3,4))
    p=PerceptualDecisionPolicy(model=None); p.spatial_control=ctrl; p.visual_tracker.observe(s,step=0)
    candidates=enumerate_decision_candidates(s,p.spatial_control,p.visual_tracker,p.affordances,p.visual_beliefs)
    ctx=p._visual_context(s,candidates)
    assert 'exploration_graph' in ctx
    assert ctx['exploration_graph']['local_untested_candidates']
    assert all('score' not in row for row in ctx['decision_candidates'])
