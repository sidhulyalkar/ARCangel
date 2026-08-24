import numpy as np

from arc3lab.memory.affordance import AffordanceMemory
from arc3lab.memory.visual_belief import VisualBeliefState
from arc3lab.perception.scene import build_scene
from arc3lab.perception.spatial import SpatialControlModel
from arc3lab.perception.visual import VisualTracker, temporal_visual_packet, visual_signature
from arc3lab.planning.counterfactual import enumerate_decision_candidates
from arc3lab.policy.visual_coding import VisualDecisionPolicy
from arc3lab.types import ActionSpec


class Frame:
    def __init__(self, grid, actions=(1, 2, 3, 4, 6), level=0, state="NOT_FINISHED"):
        self.frame = [np.asarray(grid, dtype=np.int8)]
        self.available_actions = actions
        self.levels_completed = level
        self.state = state


def scene(grid, actions=(1, 2, 3, 4, 6), level=0):
    return build_scene(Frame(grid, actions=actions, level=level))


def move(grid, src, dst):
    g = np.asarray(grid, dtype=np.int8).copy()
    color = int(g[src])
    g[src] = 0
    g[dst] = color
    return g


def train_control(base_grid):
    control = SpatialControlModel()
    cur = np.asarray(base_grid, dtype=np.int8)
    seq = [
        (1, (5, 2), (4, 2)),
        (2, (4, 2), (4, 3)),
        (3, (4, 3), (5, 3)),
        (4, (5, 3), (5, 2)),
    ]
    for aid, src, dst in seq:
        nxt = move(cur, src, dst)
        control.observe(scene(cur), scene(nxt), ActionSpec(aid))
        cur = nxt
    return control


def test_temporal_packet_contains_history_current_and_delta():
    a = np.zeros((4, 4), dtype=np.int8)
    b = a.copy(); b[1, 1] = 3
    c = b.copy(); c[1, 1] = 0; c[2, 2] = 7
    packet, meta = temporal_visual_packet([a, b, c], background=0, vanished_marker=6)
    assert packet.shape == (9, 9)
    assert packet[0, 0] == 0
    assert packet[1, 5 + 1] == 3  # top-right t-1 panel
    assert packet[5 + 2, 2] == 7  # bottom-left current panel
    assert packet[5 + 1, 5 + 1] == 6  # vanished marker in delta
    assert packet[5 + 2, 5 + 2] == 7
    assert meta["layout"]["bottom_right"].startswith("delta")


def test_visual_tracker_preserves_identity_under_translation():
    g0 = np.zeros((8, 8), dtype=np.int8); g0[4, 2] = 9
    g1 = move(g0, (4, 2), (4, 3))
    tracker = VisualTracker()
    s0, s1 = scene(g0), scene(g1)
    tracker.observe(s0, step=0)
    tid0 = tracker.track_for_component(0)
    events = tracker.observe(s1, step=1, action=ActionSpec(2))
    tid1 = tracker.track_for_component(0)
    assert tid0 == tid1
    assert any(e["kind"] == "move" and e["delta"] == [0, 1] for e in events)
    assert tracker.tracks[tid1].motion_events == 1


def test_visual_tracker_uses_nearest_match_for_duplicate_shapes():
    g0 = np.zeros((8, 8), dtype=np.int8); g0[2, 1] = 5; g0[6, 6] = 5
    g1 = np.zeros((8, 8), dtype=np.int8); g1[2, 2] = 5; g1[6, 5] = 5
    tracker = VisualTracker()
    s0, s1 = scene(g0), scene(g1)
    tracker.observe(s0, step=0)
    left_tid = tracker.track_for_component(0)
    right_tid = tracker.track_for_component(1)
    tracker.observe(s1, step=1, action=ActionSpec(1))
    assert tracker.tracks[left_tid].last_center == (2, 2)
    assert tracker.tracks[right_tid].last_center == (6, 5)


def test_affordance_memory_learns_click_target_effect():
    g0 = np.zeros((8, 8), dtype=np.int8); g0[2, 3] = 4
    g1 = np.zeros((8, 8), dtype=np.int8)
    s0, s1 = scene(g0), scene(g1)
    mem = AffordanceMemory()
    action = ActionSpec(6, x=3, y=2)
    effect = mem.observe(s0, s1, action)
    sig = visual_signature(s0.components[0])
    assert effect == "remove"
    p = mem.click_posterior(sig)
    assert p.support == 1
    assert p.counts["remove"] == 1
    candidates = mem.click_candidates(s0, VisualTracker(), max_candidates=8)
    assert candidates and candidates[0]["x"] == 3 and candidates[0]["y"] == 2


def test_visual_goal_belief_tracks_signature_not_absolute_position():
    g0 = np.zeros((8, 8), dtype=np.int8); g0[2, 2] = 3
    g1 = np.zeros((8, 8), dtype=np.int8); g1[5, 6] = 3
    s0, s1 = scene(g0), scene(g1)
    tracker = VisualTracker(); tracker.observe(s0, step=0)
    beliefs = VisualBeliefState()
    beliefs.update_from_model(
        [{"target_object": 0, "relation": "click", "confidence": 0.8, "evidence": "unique persistent object"}],
        s0,
        tracker,
        level=0,
    )
    current = beliefs.top_current(s1)
    assert current and current[0]["target_object"] == 0
    assert current[0]["relation"] == "click"


def test_counterfactual_move_toward_goal_beats_move_away():
    g = np.zeros((10, 10), dtype=np.int8)
    g[5, 2] = 7  # actor
    g[5, 8] = 3  # target
    s = scene(g, actions=(1, 2, 3, 4))
    control = train_control(g)
    tracker = VisualTracker(); tracker.observe(s, step=0)
    beliefs = VisualBeliefState()
    target_index = next(i for i, c in enumerate(s.components) if c.color == 3)
    beliefs.update_from_model(
        [{"target_object": target_index, "relation": "touch", "confidence": 0.95, "evidence": "target"}],
        s, tracker, level=0,
    )
    aff = AffordanceMemory()
    candidates = enumerate_decision_candidates(s, control, tracker, aff, beliefs)
    primitive = [c for c in candidates if c.kind == "primitive"]
    by_delta = {tuple(c.payload["movement"]["delta"]): c for c in primitive if c.payload["movement"]}
    assert by_delta[(0, 1)].score > by_delta[(0, -1)].score


def test_counterfactual_exposes_spatial_plan_for_supported_goal():
    g = np.zeros((10, 10), dtype=np.int8)
    g[5, 2] = 7
    g[5, 8] = 3
    control = train_control(g)
    s = scene(g, actions=(1, 2, 3, 4))
    tracker = VisualTracker(); tracker.observe(s, step=0)
    beliefs = VisualBeliefState()
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    beliefs.update_from_model(
        [{"target_object": target, "relation": "touch", "confidence": 0.9, "evidence": "target"}],
        s, tracker, level=0,
    )
    candidates = enumerate_decision_candidates(s, control, tracker, AffordanceMemory(), beliefs)
    plans = [c for c in candidates if c.kind == "spatial_plan"]
    assert plans and plans[0].payload["steps"] > 0


def test_game_over_evidence_penalizes_action_candidate():
    g = np.zeros((8, 8), dtype=np.int8); g[4, 2] = 7; g[4, 6] = 3
    s = scene(g, actions=(1, 2))
    control = SpatialControlModel()
    tracker = VisualTracker(); tracker.observe(s, step=0)
    beliefs = VisualBeliefState(); aff = AffordanceMemory()
    # Give action 1 repeated game-over evidence and action 2 repeated benign changes.
    for _ in range(4):
        aff.action_effects[1]["game_over"] += 1
        aff.action_effects[2]["change"] += 1
    candidates = enumerate_decision_candidates(s, control, tracker, aff, beliefs)
    prim = {c.spec.action_id: c for c in candidates if c.kind == "primitive"}
    assert prim[2].score > prim[1].score


class CandidateModel:
    def __init__(self):
        self.calls = 0
        self.grids = []
        self.prompts = []

    def complete(self, system, user, grid=None):
        self.calls += 1
        self.grids.append(np.asarray(grid))
        self.prompts.append(user)
        return (
            '{"hypothesis":"use exact candidate registry","goal":"test safe move",'
            '"visual_goals":[],"orientation":{"controlled":"uncertain"},'
            '"decision_mode":"probe","candidate_id":"a0","python":"",'
            '"analysis_question":"","actions":[],"confidence":0.8,'
            '"plan_reliable":false,"expected_change":"observe causal effect",'
            '"delegate_world_model":false}'
        )


def test_visual_policy_sends_temporal_packet_and_selects_registered_candidate():
    model = CandidateModel()
    policy = VisualDecisionPolicy(model=model, max_model_calls=2, max_tool_calls=0)
    g = np.zeros((8, 8), dtype=np.int8); g[4, 3] = 2
    s = scene(g, actions=(1, 2))
    # Seed visual tracker because this unit test invokes _reason directly rather than observe().
    policy.visual_tracker.observe(s, step=0)
    policy.grids.extend([g.copy(), g.copy(), g.copy()])
    actions = policy._reason(s)
    assert actions and actions[0].action_id in {1, 2}
    assert model.calls == 1
    assert model.grids[0].shape == (17, 17)
    assert "TEMPORAL VISUAL COGNITION" in model.prompts[0]
    assert "decision_candidates" in model.prompts[0]
    assert policy.visual_candidate_selections == 1


def test_visual_policy_goal_output_becomes_executable_belief():
    policy = VisualDecisionPolicy(model=None)
    g = np.zeros((8, 8), dtype=np.int8); g[2, 2] = 4
    s = scene(g)
    policy.visual_tracker.observe(s, step=0)
    policy._remember_visual_parse(
        {
            "visual_goals": [
                {"target_object": 0, "relation": "click", "confidence": 0.87, "evidence": "isolated visual token"}
            ],
            "orientation": {"important": [0]},
            "decision_mode": "probe",
        },
        s,
    )
    summary = policy.visual_beliefs.summary(s)
    assert summary["current_goal_candidates"][0]["confidence"] >= 0.87
    assert policy.last_decision_mode == "probe"

def test_visual_tracker_surfaces_multicolor_comoving_entity_parts():
    g0 = np.zeros((8, 8), dtype=np.int8)
    g0[4, 2] = 2
    g0[4, 3] = 3
    g0[1, 6] = 8
    g1 = np.zeros_like(g0)
    g1[3, 2] = 2
    g1[3, 3] = 3
    g1[1, 6] = 8
    tracker = VisualTracker()
    s0, s1 = scene(g0), scene(g1)
    tracker.observe(s0, step=0)
    tracker.observe(s1, step=1, action=ActionSpec(1))
    groups = tracker.summary(s1)["co_moving_groups"]
    assert groups
    assert groups[0]["delta"] == [-1, 0]
    assert groups[0]["parts"] == 2


def test_local_adapter_supports_higher_resolution_visual_packet():
    from arc3lab.model.adapter import OpenAICompatLocalAdapter
    assert OpenAICompatLocalAdapter(image_side=384).image_side == 384
    assert OpenAICompatLocalAdapter(image_side=9999).image_side == 768
    assert OpenAICompatLocalAdapter(image_side=32).image_side == 128
