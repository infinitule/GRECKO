/**
 * Canvas-based air picture renderer.
 *
 * Coordinate system: world metres → canvas pixels via a configurable
 * viewport.  Y is flipped so world north is canvas top.
 *
 * Renders:
 *   - Asset (white diamond)
 *   - Tracks with 1-sigma uncertainty ellipses (colour by threat label)
 *   - Intent cluster outlines (doctrine colour rings)
 *   - Assignment lines (green=authorized, amber=pending, grey=hold)
 *   - Interceptors (cyan triangles)
 *   - Comms mesh edges (dim blue)
 *   - Truth entities (dim ghosts, optional)
 *   - Forecast centroid paths (dashed)
 */
import type { SimState, TrackInfo } from "./types.js";

const COLOURS = {
  hostile:   "#e03030",
  unknown:   "#e8a020",
  friendly:  "#28c860",
  tentative: "#5a6a7a",
  authorized:"#28c860",
  pending:   "#e8a020",
  hold:      "#505878",
  asset:     "#e0e0ff",
  interceptor: "#00d8ff",
  comms:     "rgba(32,128,232,0.25)",
  truth:     "rgba(180,180,255,0.20)",
  main_axis: "#e03030",
  feint:     "#e87820",
  isr_loiter:"#3888e0",
  reserve:   "#888899",
  frontal_saturation: "#e03030",
  leader_follower: "#aa44cc",
} as const;

interface Viewport {
  cx: number;   // world x at canvas centre
  cy: number;   // world y at canvas centre
  scale: number; // pixels per metre
}

export class AirPicture {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private vp: Viewport = { cx: 0, cy: 0, scale: 0.18 };
  private showTruth = false;
  private showComms = true;
  private showIntent = true;
  private lastState: SimState | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this._bindResize();
    this._bindDrag();

    // Option toggles
    document.getElementById("show-truth")?.addEventListener("change", (e) => {
      this.showTruth = (e.target as HTMLInputElement).checked;
      if (this.lastState) this.render(this.lastState);
    });
    document.getElementById("show-comms")?.addEventListener("change", (e) => {
      this.showComms = (e.target as HTMLInputElement).checked;
      if (this.lastState) this.render(this.lastState);
    });
    document.getElementById("show-intent")?.addEventListener("change", (e) => {
      this.showIntent = (e.target as HTMLInputElement).checked;
      if (this.lastState) this.render(this.lastState);
    });
  }

  render(state: SimState): void {
    this.lastState = state;
    const { canvas, ctx } = this;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = "#080c12";
    ctx.fillRect(0, 0, W, H);

    // Grid
    this._drawGrid();

    // Comms mesh
    if (this.showComms) {
      this._drawCommsMesh(state);
    }

    // Truth entities (optional debug layer)
    if (this.showTruth) {
      for (const e of state.truth_entities) {
        const [cx, cy] = this._toCanvas(e.pos[0], e.pos[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = COLOURS.truth;
        ctx.fill();
      }
    }

    // Intent cluster rings (behind tracks)
    if (this.showIntent) {
      this._drawIntentRings(state);
    }

    // Assignment lines
    this._drawAssignmentLines(state);

    // Tracks
    for (const trk of state.tracks) {
      this._drawTrack(trk, state);
    }

    // Interceptors
    for (const iv of state.interceptors) {
      if (!iv.alive) continue;
      this._drawInterceptor(iv.pos, iv.heading, iv.assigned_track !== null);
    }

    // Asset
    this._drawAsset(state.asset.pos, state.asset.hp);
  }

  // ------------------------------------------------------------------ //

  private _toCanvas(wx: number, wy: number): [number, number] {
    const { cx, cy, scale } = this.vp;
    const W = this.canvas.width;
    const H = this.canvas.height;
    return [
      W / 2 + (wx - cx) * scale,
      H / 2 - (wy - cy) * scale,  // flip Y
    ];
  }

  private _drawGrid(): void {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const step = 200; // metres
    ctx.strokeStyle = "rgba(30,39,54,0.8)";
    ctx.lineWidth = 0.5;
    const startX = Math.floor((this.vp.cx - W / 2 / this.vp.scale) / step) * step;
    const startY = Math.floor((this.vp.cy - H / 2 / this.vp.scale) / step) * step;
    for (let wx = startX; wx < startX + W / this.vp.scale + step * 2; wx += step) {
      const [px] = this._toCanvas(wx, 0);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, H);
      ctx.stroke();
    }
    for (let wy = startY; wy < startY + H / this.vp.scale + step * 2; wy += step) {
      const [, py] = this._toCanvas(0, wy);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(W, py);
      ctx.stroke();
    }
  }

  private _drawCommsMesh(state: SimState): void {
    const ctx = this.ctx;
    // Build position map from interceptors + asset
    const pos: Record<string, [number, number]> = {};
    for (const iv of state.interceptors) {
      pos[iv.id] = iv.pos;
    }
    pos["C2"] = state.asset.pos;

    ctx.strokeStyle = COLOURS.comms;
    ctx.lineWidth = 0.8;
    for (const [a, b] of state.mesh_topology.edges) {
      const pa = pos[a];
      const pb = pos[b];
      if (!pa || !pb) continue;
      const [ax, ay] = this._toCanvas(pa[0], pa[1]);
      const [bx, by] = this._toCanvas(pb[0], pb[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  }

  private _drawIntentRings(state: SimState): void {
    const ctx = this.ctx;
    for (const intent of state.intents) {
      const memberTracks = intent.member_track_ids
        .map(id => state.tracks.find(t => t.track_id === id))
        .filter(Boolean) as TrackInfo[];
      if (memberTracks.length === 0) continue;

      const colour = (COLOURS as Record<string, string>)[intent.dominant_intent] ?? COLOURS.reserve;
      ctx.strokeStyle = colour + "66";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);

      for (const trk of memberTracks) {
        const [cx, cy] = this._toCanvas(trk.pos[0], trk.pos[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Forecast centroid path
      if (intent.forecast_centroids.length > 0) {
        ctx.strokeStyle = colour + "44";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 6]);
        ctx.beginPath();
        const [fx0, fy0] = this._toCanvas(
          intent.forecast_centroids[0][0], intent.forecast_centroids[0][1]
        );
        ctx.moveTo(fx0, fy0);
        for (let i = 1; i < Math.min(intent.forecast_centroids.length, 5); i++) {
          const [fx, fy] = this._toCanvas(
            intent.forecast_centroids[i][0], intent.forecast_centroids[i][1]
          );
          ctx.lineTo(fx, fy);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }

  private _drawAssignmentLines(state: SimState): void {
    const ctx = this.ctx;
    // Build track_id → estimated position map
    const trackPos: Record<string, [number, number]> = {};
    for (const trk of state.tracks) {
      trackPos[trk.track_id] = trk.pos;
    }
    for (const a of state.assignments) {
      if (a.action !== "ASSIGN" || !a.track_id) continue;
      const iv = state.interceptors.find(i => i.id === a.interceptor_id);
      const tp = trackPos[a.track_id];
      if (!iv || !tp) continue;

      const isAuth = state.authorized_tracks.includes(a.track_id);
      const isHeld = state.held_tracks.includes(a.track_id);

      let colour = COLOURS.hold;
      if (isAuth) colour = COLOURS.authorized + "aa";
      else if (!isHeld) colour = COLOURS.pending + "aa";

      const [ix, iy] = this._toCanvas(iv.pos[0], iv.pos[1]);
      const [tx, ty] = this._toCanvas(tp[0], tp[1]);
      ctx.beginPath();
      ctx.moveTo(ix, iy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  private _drawTrack(trk: TrackInfo, state: SimState): void {
    const ctx = this.ctx;
    const assess = state.assessments[trk.track_id];
    const label = assess?.label ?? "unknown";
    const conf = assess?.confidence ?? 0.5;

    const colour = trk.status === "tentative"
      ? COLOURS.tentative
      : (COLOURS as Record<string, string>)[label] ?? COLOURS.unknown;

    const [cx, cy] = this._toCanvas(trk.pos[0], trk.pos[1]);

    // 1-sigma uncertainty ellipse
    const sigx = Math.sqrt(trk.cov_diag[0]) * this.vp.scale;
    const sigy = Math.sqrt(trk.cov_diag[1]) * this.vp.scale;
    const rx = Math.max(sigx, 3);
    const ry = Math.max(sigy, 3);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = colour + "55";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Track dot
    const r = trk.status === "confirmed" ? 5 : 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = colour;
    ctx.globalAlpha = 0.4 + conf * 0.6;
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Velocity vector
    const vscale = 3.0;
    const [vx, vy] = [trk.vel[0] * vscale * this.vp.scale,
                      -trk.vel[1] * vscale * this.vp.scale];
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + vx, cy + vy);
    ctx.strokeStyle = colour + "88";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    ctx.font = "9px monospace";
    ctx.fillStyle = colour;
    ctx.fillText(trk.track_id.slice(-4), cx + 6, cy - 4);

    // Authorization indicator
    if (state.authorized_tracks.includes(trk.track_id)) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = COLOURS.authorized + "99";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  private _drawInterceptor(pos: [number, number], heading: number, engaged: boolean): void {
    const ctx = this.ctx;
    const [cx, cy] = this._toCanvas(pos[0], pos[1]);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-heading + Math.PI / 2);  // triangle points up = north
    ctx.beginPath();
    ctx.moveTo(0, -8);
    ctx.lineTo(5, 6);
    ctx.lineTo(-5, 6);
    ctx.closePath();
    ctx.strokeStyle = COLOURS.interceptor;
    ctx.lineWidth = engaged ? 2 : 1;
    ctx.stroke();
    ctx.restore();
  }

  private _drawAsset(pos: [number, number], hp: number): void {
    const ctx = this.ctx;
    const [cx, cy] = this._toCanvas(pos[0], pos[1]);
    const hpFrac = Math.max(0, hp / 10);
    ctx.fillStyle = `rgba(${Math.round(255 * (1 - hpFrac))},${Math.round(220 * hpFrac)},${Math.round(255 * hpFrac)},0.9)`;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx + 10, cy);
    ctx.lineTo(cx, cy + 10);
    ctx.lineTo(cx - 10, cy);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = COLOURS.asset;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = "9px monospace";
    ctx.fillStyle = COLOURS.asset;
    ctx.fillText(`HP:${hp.toFixed(0)}`, cx + 12, cy + 3);
  }

  // ------------------------------------------------------------------ //
  // Interaction                                                          //
  // ------------------------------------------------------------------ //

  private _bindResize(): void {
    const resize = () => {
      const rect = this.canvas.parentElement!.getBoundingClientRect();
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
      if (this.lastState) this.render(this.lastState);
    };
    window.addEventListener("resize", resize);
    resize();
  }

  private _bindDrag(): void {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    this.canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const dx = (e.clientX - lastX) / this.vp.scale;
      const dy = (e.clientY - lastY) / this.vp.scale;
      this.vp.cx -= dx;
      this.vp.cy += dy;
      lastX = e.clientX;
      lastY = e.clientY;
      if (this.lastState) this.render(this.lastState);
    });
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      this.vp.scale = Math.max(0.05, Math.min(2.0, this.vp.scale * factor));
      if (this.lastState) this.render(this.lastState);
    }, { passive: false });
  }
}
