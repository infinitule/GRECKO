/** Shared types for the AEGISNET C2 WebSocket protocol. */

export interface TrackInfo {
  track_id: string;
  status: "tentative" | "confirmed" | "coasted";
  pos: [number, number];
  vel: [number, number];
  cov_diag: [number, number]; // [σx², σy²]
  quality: number;
  age: number;
  n_updates: number;
}

export interface AssessmentInfo {
  label: "hostile" | "unknown" | "friendly";
  confidence: number;
  priority_score: number;
  why: string;
}

export interface AssignmentInfo {
  interceptor_id: string;
  action: "ASSIGN" | "HOLD_FIRE" | "RTB";
  track_id: string | null;
  effector_id: string | null;
  bid_value: number;
  hold_reason: string;
}

export interface IntentInfo {
  cluster_id: string;
  member_track_ids: string[];
  intent_distribution: Record<string, number>;
  dominant_intent: string;
  value_multiplier: number;
  forecast_centroids: [number, number][];
}

export interface InterceptorInfo {
  id: string;
  pos: [number, number];
  vel: [number, number];
  heading: number;
  alive: boolean;
  endurance: number;
  effector_type: string;
  assigned_track: string | null;
}

export interface TruthEntityInfo {
  id: string;
  pos: [number, number];
  vel: [number, number];
  alive: boolean;
}

export interface AssetInfo {
  id: string;
  pos: [number, number];
  hp: number;
  alive: boolean;
}

export interface MeshTopology {
  edges: [string, string][];
  partitions: string[][];
  partition_count: number;
}

export interface AuditEntry {
  sim_t: number;
  wall_t: number;
  event: string;
  interceptor_id: string;
  track_id: string;
  detail: string;
}

export interface SimState {
  t: number;
  tracks: TrackInfo[];
  assessments: Record<string, AssessmentInfo>;
  assignments: AssignmentInfo[];
  intents: IntentInfo[];
  interceptors: InterceptorInfo[];
  truth_entities: TruthEntityInfo[];
  asset: AssetInfo;
  mesh_topology: MeshTopology;
  magazine: Record<string, number>;
  weapons_hold: boolean;
  authorized_tracks: string[];
  held_tracks: string[];
  friendly_tracks: string[];
  lambda_cost: number;
  audit_trail: AuditEntry[];
}

// ------------------------------------------------------------------ //
// C2 commands (client → server)                                       //
// ------------------------------------------------------------------ //

export type C2Command =
  | { type: "AUTHORIZE"; track_id: string }
  | { type: "HOLD"; track_id: string }
  | { type: "MARK_FRIENDLY"; track_id: string }
  | { type: "LIFT_HOLD"; track_id: string }
  | { type: "WEAPONS_HOLD"; active: boolean }
  | { type: "SET_LAMBDA"; value: number }
  | { type: "PAUSE" }
  | { type: "PLAY" }
  | { type: "SET_SPEED"; value: number }
  | { type: "PING" };

// ------------------------------------------------------------------ //
// Server messages                                                     //
// ------------------------------------------------------------------ //

export type ServerMessage =
  | (SimState & { type?: undefined })
  | { type: "PONG"; t: number; wall: number }
  | { type: "SIM_END"; t: number; summary: Record<string, unknown>; log_hash: string }
  | { type: "ERROR"; msg: string };
