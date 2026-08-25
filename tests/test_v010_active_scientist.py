import json
import numpy as np

from arc3lab.perception.scene import build_scene
from arc3lab.planning.counterfactual import enumerate_decision_candidates
from arc3lab.policy.active_scientist import ACTIVE_SCIENTIST_SYSTEM_PROMPT, ActiveScientistPolicy
from arc3lab.types import ActionSpec


class Frame:
    def __init__(self, grid, actions=(1, 2, 3, 4, 6), level=0, state="NOT_FINISHED"):
        self.frame = [np.asarray(grid, dtype=np.int8)]
        self.available_actions = actions
        self.levels_completed = level
        self.state = state


class FakeModel:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
    def complete(self, system, user, *, grid=None):
        self.calls.append((system, user, grid))
        return json.dumps(self.payloads.pop(0))


def scene(grid=None, actions=(1, 2, 3, 4, 6)):
    if grid is None:
        grid = np.zeros((8, 8), dtype=np.int8); grid[4, 2] = 7; grid[1, 6] = 3
    return build_scene(Frame(grid, actions), step=0)


def base_payload(**kwargs):
    p = {
        "orientation":{"controlled_entity":"uncertain","important_changes":[],"salient_objects":["object 0"],"dominant_uncertainty":"agency"},
        "hypotheses":{"agency":[],"mechanics":[],"goals":[],"abstraction":[]},
        "visual_goals":[], "decision_mode":"TEST_HYPOTHESIS", "candidate_id":"a0", "actions":[],
        "hypothesis_confidence":0.2, "action_confidence":0.9,
        "experiment":{"question":"does ACTION1 control the isolated object?","distinguishes":["controlled","not controlled"],"expected_outcomes":["motion","no motion"]},
        "python":"", "analysis_question":"", "delegate_world_model":False, "plan_reliable":False,
        "expected_change":"one persistent object moves", "memory_note":"", "goal":"", "hypothesis":"ACTION1 may control one object",
    }
    p.update(kwargs); return p


def test_system_prompt_has_one_canonical_confidence_contract():
    assert "hypothesis_confidence" in ACTIVE_SCIENTIST_SYSTEM_PROMPT
    assert "action_confidence" in ACTIVE_SCIENTIST_SYSTEM_PROMPT
    assert "Return exactly one JSON object" in ACTIVE_SCIENTIST_SYSTEM_PROMPT


def test_low_theory_confidence_does_not_block_high_value_experiment():
    policy = ActiveScientistPolicy(model=FakeModel([base_payload(hypothesis_confidence=.12, action_confidence=.94)]), max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(scene().grid)))
    assert action.action_id == 1 and action.confidence == .94
    assert policy.semantic_actions == 1 and policy.emergency_fallback_actions == 0


def test_model_directed_frontier_is_not_emergency_fallback():
    policy = ActiveScientistPolicy(model=FakeModel([base_payload(decision_mode="EXPLORE_FRONTIER", candidate_id="a1")]), max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(scene().grid)))
    assert action.action_id == 2
    assert policy.model_directed_frontier_actions == 1 and policy.emergency_fallback_actions == 0


def test_low_action_confidence_rejects_without_candidate_side_effects():
    policy = ActiveScientistPolicy(model=FakeModel([base_payload(action_confidence=.05)]), max_model_calls=4, max_tool_calls=0)
    policy.choose(policy.observe(Frame(scene().grid, actions=(1,2,3,4))))
    assert policy.low_action_confidence_rejections == 1
    assert policy.semantic_actions == 0 and policy.emergency_fallback_actions == 1
    assert policy.visual_candidate_selections == 0 and not policy._planned_steps


def test_emergency_fallback_does_not_pollute_semantic_selection():
    policy = ActiveScientistPolicy(model=None)
    policy.choose(policy.observe(Frame(scene().grid, actions=(1,2,3,4))))
    assert policy.semantic_actions == 0 and policy.emergency_fallback_actions == 1
    assert policy.visual_candidate_selections == 0


def test_visual_goal_proposal_becomes_executable_belief():
    payload = base_payload(visual_goals=[{"target_object":1,"relation":"touch","confidence":.78,"evidence":"isolated target"}])
    policy = ActiveScientistPolicy(model=FakeModel([payload]), max_model_calls=4, max_tool_calls=0)
    policy.choose(policy.observe(Frame(scene().grid)))
    assert policy.goal_proposals == 1 and policy.visual_goal_updates >= 1
    assert policy.visual_beliefs.beliefs[0].relation == "touch"


def test_direct_action_uses_action_confidence_not_theory_confidence():
    payload = base_payload(candidate_id="", actions=[{"id":3,"x":None,"y":None}], hypothesis_confidence=.11, action_confidence=.88)
    policy = ActiveScientistPolicy(model=FakeModel([payload]), max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(scene().grid, actions=(1,2,3,4))))
    assert action.action_id == 3 and action.confidence == .88 and policy.semantic_direct_actions == 1


def test_bad_or_incomplete_contract_is_counted():
    policy = ActiveScientistPolicy(model=None)
    parsed = policy._normalize({"decision_mode":"IDENTIFY_AGENCY","hypothesis_confidence":.2,"action_confidence":.8,"actions":[{"id":1}]})
    assert parsed["decision_mode"] == "TEST_HYPOTHESIS" and not parsed["contract_valid"]
    parsed = policy._normalize({"decision_mode":"ACT_DIRECTLY","actions":[{"id":1}]})
    assert not parsed["contract_valid"] and policy.model_parse_contract_errors == 2


def test_agency_fallback_replicates_candidate_motion():
    g0=np.zeros((9,9),dtype=np.int8); g0[5,2]=7; g0[1,7]=3
    g1=g0.copy(); g1[5,2]=0; g1[4,2]=7
    s0,s1=scene(g0,(1,2,3,4)),scene(g1,(1,2,3,4))
    policy=ActiveScientistPolicy(model=None); policy.spatial_control.observe(s0,s1,ActionSpec(1)); policy.last_perceptual_state={"recommended_mode":"IDENTIFY_AGENCY"}
    candidates=enumerate_decision_candidates(s1,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs)
    c,why=policy._mode_fallback(s1,candidates)
    assert c.spec.action_id == 1 and why == "agency_replicate_motion"


def test_repair_mode_retests_same_primitive():
    sc=scene(actions=(1,2,3,4)); policy=ActiveScientistPolicy(model=None); policy.last_action=ActionSpec(3); policy.last_perceptual_state={"recommended_mode":"REPAIR_MODEL"}
    c,why=policy._mode_fallback(sc,enumerate_decision_candidates(sc,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs))
    assert c.spec.action_id == 3 and why == "repair_retest"


def test_phase_mode_prefers_observed_quasi_noop():
    g=np.zeros((8,8),dtype=np.int8); g[4,2]=7; sc=scene(g,(1,2,3,4)); policy=ActiveScientistPolicy(model=None)
    policy.affordances.observe(sc,sc,ActionSpec(2)); policy.last_perceptual_state={"recommended_mode":"REASON_ABOUT_PHASE"}
    c,why=policy._mode_fallback(sc,enumerate_decision_candidates(sc,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs))
    assert c.spec.action_id == 2 and why == "phase_quasi_noop"


def test_trace_and_semantic_telemetry_are_structured_not_chain_of_thought():
    policy=ActiveScientistPolicy(model=FakeModel([base_payload()]), max_model_calls=4, max_tool_calls=0)
    policy.choose(policy.observe(Frame(scene().grid, actions=(1,2,3,4))))
    trace=policy.last_decision_trace; tel=policy.semantic_telemetry()
    assert trace["model_action_accepted"] is True and "experiment" in trace and "chain_of_thought" not in trace
    assert tel["semantic_actions"] == 1 and tel["model_parse_successes"] == 1


def test_query_history_roundtrip_can_produce_semantic_action():
    first=base_payload(decision_mode="QUERY_HISTORY",candidate_id="",actions=[],python="result = action_stats()",action_confidence=.9)
    second=base_payload(candidate_id="a2",action_confidence=.91)
    policy=ActiveScientistPolicy(model=FakeModel([first,second]), max_model_calls=4, max_tool_calls=2)
    action=policy.choose(policy.observe(Frame(scene().grid,actions=(1,2,3,4))))
    assert action.action_id == 3 and policy.tool_calls == 1 and policy.model_calls == 2 and policy.semantic_actions == 1


def test_fallback_reasons_are_mode_specific():
    policy=ActiveScientistPolicy(model=None); sc=scene(actions=(1,2,3,4)); candidates=enumerate_decision_candidates(sc,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs)
    for mode,fragment in [("IDENTIFY_AGENCY","agency_"),("IDENTIFY_GOAL","identify_goal_"),("DISCRIMINATE_DYNAMICS","discriminate_dynamics_"),("MODEL_AND_TEST","model_and_test_")]:
        policy.last_perceptual_state={"recommended_mode":mode}; c,why=policy._mode_fallback(sc,candidates); assert c is not None and fragment in why


def test_active_user_prompt_does_not_reintroduce_legacy_schema():
    from arc3lab.policy.active_scientist import ACTIVE_USER_TEMPLATE
    assert "OUTPUT SCHEMA" not in ACTIVE_USER_TEMPLATE and "V010 DECISION SCHEMA" in ACTIVE_USER_TEMPLATE


def test_typed_hypotheses_persist_and_contradictions_reduce_confidence():
    policy=ActiveScientistPolicy(model=None)
    assert policy.hypothesis_registry.update_from_model({"mechanics":[{"statement":"ACTION2 translates actor east","confidence":.8,"evidence":"test"}]},level=0,step=3)==1
    before=policy.hypothesis_registry.summary()["mechanics"][0]
    assert policy.hypothesis_registry.contradict_recent(level=0,step=4)==1
    after=policy.hypothesis_registry.summary()["mechanics"][0]
    assert after["confidence"] < before["confidence"]


def test_semantic_goal_compiles_exact_spatial_plan():
    g=np.zeros((10,10),dtype=np.int8); g[5,2]=7; g[5,8]=3; policy=ActiveScientistPolicy(model=None); cur=g.copy()
    for aid,src,dst in [(1,(5,2),(4,2)),(2,(4,2),(4,3)),(3,(4,3),(5,3)),(4,(5,3),(5,2))]:
        nxt=cur.copy(); nxt[src]=0; nxt[dst]=7; policy.spatial_control.observe(scene(cur,(1,2,3,4)),scene(nxt,(1,2,3,4)),ActionSpec(aid)); cur=nxt
    sc=scene(g,(1,2,3,4)); policy.visual_tracker.observe(sc,step=0); target=next(i for i,c in enumerate(sc.components) if c.color==3)
    policy.visual_beliefs.update_from_model([{"target_object":target,"relation":"touch","confidence":.9,"evidence":"test"}],sc,policy.visual_tracker,level=0)
    candidates=enumerate_decision_candidates(sc,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs); plan=next(c for c in candidates if c.kind=="spatial_plan")
    parsed=policy._normalize({"decision_mode":"EXECUTE_VERIFIED_PLAN","candidate_id":plan.candidate_id,"actions":[],"hypothesis_confidence":.75,"action_confidence":.92,"visual_goals":[]})
    assert policy._model_action_from_parse(parsed,sc,candidates) and policy.spatial_plans_compiled==1 and policy.semantic_actions==1


def test_end_to_end_scientist_transitions_from_probes_to_plan_and_queue():
    class CopyFrame(Frame):
        def __init__(self,grid):
            self.frame=[np.array(grid,dtype=np.int8,copy=True)]; self.available_actions=(1,2,3,4); self.levels_completed=0; self.state="NOT_FINISHED"
    class ScriptedModel:
        def __init__(self): self.calls=0
        def complete(self,system,user,*,grid=None):
            self.calls+=1
            if self.calls<=4: return json.dumps(base_payload(candidate_id=f"a{self.calls-1}",action_confidence=.95))
            return json.dumps(base_payload(candidate_id="p0",decision_mode="EXECUTE_VERIFIED_PLAN",hypothesis_confidence=.8,action_confidence=.95,visual_goals=[{"target_object":0,"relation":"touch","confidence":.9,"evidence":"isolated target"}]))
    g=np.zeros((24,24),dtype=np.int8); g[9,18]=3; g[13,10]=7; pos=(13,10); moves={1:(-1,0),2:(0,1),3:(1,0),4:(0,-1)}; model=ScriptedModel(); policy=ActiveScientistPolicy(model=model,max_model_calls=20,max_tool_calls=4,spatial_plan_horizon=24)
    for _ in range(7):
        action=policy.choose(policy.observe(CopyFrame(g))); dr,dc=moves[action.action_id]; nr,nc=pos[0]+dr,pos[1]+dc; ng=g.copy(); ng[pos]=0; ng[nr,nc]=7; g,pos=ng,(nr,nc)
    assert policy.spatial_plans_compiled==1 and policy.semantic_actions==5 and policy.queued_actions_used==2 and model.calls==5


def test_repair_action6_never_switches_to_different_target():
    g=np.zeros((8,8),dtype=np.int8); g[2,2]=4; g[5,5]=5; sc=scene(g,actions=(6,)); policy=ActiveScientistPolicy(model=None); policy.visual_tracker.observe(sc,step=0); policy.last_action=ActionSpec(6,x=5,y=5); policy.last_perceptual_state={"recommended_mode":"REPAIR_MODEL"}
    c,why=policy._mode_fallback(sc,enumerate_decision_candidates(sc,policy.spatial_control,policy.visual_tracker,policy.affordances,policy.visual_beliefs))
    if c is not None: assert (c.spec.x,c.spec.y)==(5,5) and why=="repair_retest"
