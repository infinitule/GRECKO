/**
 * Human-on-the-loop panel.
 *
 * Renders one row per confirmed/coasted track with three action buttons:
 *   AUTHORIZE — forward the track to the allocator as engageable
 *   HOLD      — explicitly block engagement on this track
 *   FRIENDLY  — mark the track as friendly (classifier caps to unknown)
 *
 * Also controls the global WEAPONS HOLD button.
 */
import type { C2WebSocket } from "./ws.js";
import type { SimState } from "./types.js";

export class HOTLPanel {
  private ws: C2WebSocket;
  private container: HTMLElement;
  private weaponsHoldBtn: HTMLButtonElement;
  private authCountEl: HTMLElement;

  constructor(ws: C2WebSocket) {
    this.ws = ws;
    this.container = document.getElementById("hotl-tracks-container")!;
    this.weaponsHoldBtn = document.getElementById("weapons-hold-btn") as HTMLButtonElement;
    this.authCountEl = document.getElementById("auth-count")!;
    this._bindWeaponsHold();
  }

  render(state: SimState): void {
    // Update weapons hold button state
    const isHold = state.weapons_hold;
    this.weaponsHoldBtn.classList.toggle("hold-active", isHold);
    this.weaponsHoldBtn.textContent = isHold ? "⚠ WEAPONS HOLD ACTIVE" : "WEAPONS HOLD";
    document.getElementById("weapons-hold-banner")?.classList.toggle("active", isHold);

    // Auth count
    const nAuth = state.authorized_tracks.length;
    const nTracks = state.tracks.filter(t => t.status !== "tentative").length;
    this.authCountEl.textContent = `${nAuth}/${nTracks} auth`;

    // Build track rows
    const visibleTracks = state.tracks
      .filter(t => t.status !== "tentative")
      .sort((a, b) => {
        const pa = state.assessments[a.track_id]?.priority_score ?? 0;
        const pb = state.assessments[b.track_id]?.priority_score ?? 0;
        return pb - pa;
      });

    // Reconcile DOM — update existing rows or add/remove
    const existingRows = new Map<string, HTMLElement>();
    this.container.querySelectorAll<HTMLElement>("[data-track-id]").forEach(el => {
      existingRows.set(el.dataset["trackId"]!, el);
    });

    const keepIds = new Set(visibleTracks.map(t => t.track_id));

    // Remove stale rows
    for (const [id, el] of existingRows) {
      if (!keepIds.has(id)) el.remove();
    }

    // Update or create rows
    for (const trk of visibleTracks) {
      const assess = state.assessments[trk.track_id];
      const label = assess?.label ?? "unknown";
      const conf = assess?.confidence ?? 0;
      const isAuth = state.authorized_tracks.includes(trk.track_id);
      const isHeld = state.held_tracks.includes(trk.track_id);
      const isFriendly = state.friendly_tracks.includes(trk.track_id);

      let row = existingRows.get(trk.track_id);
      if (!row) {
        row = this._createRow(trk.track_id);
        this.container.appendChild(row);
      }

      row.className = `track-row ${label}`;
      (row.querySelector(".track-id") as HTMLElement).textContent = trk.track_id;
      (row.querySelector(".label") as HTMLElement).textContent = label.toUpperCase();
      (row.querySelector(".confidence") as HTMLElement).textContent =
        `${(conf * 100).toFixed(0)}%`;

      const authBtn = row.querySelector<HTMLButtonElement>(".btn.auth");
      const holdBtn = row.querySelector<HTMLButtonElement>(".btn.hold");
      const frBtn   = row.querySelector<HTMLButtonElement>(".btn.friendly-btn");
      if (authBtn) authBtn.classList.toggle("active", isAuth);
      if (holdBtn) holdBtn.classList.toggle("active", isHeld);
      if (frBtn)   frBtn.classList.toggle("active", isFriendly);
    }
  }

  private _createRow(trackId: string): HTMLElement {
    const row = document.createElement("div");
    row.className = "track-row";
    row.dataset["trackId"] = trackId;
    row.innerHTML = `
      <span class="track-id"></span>
      <span class="label"></span>
      <span class="confidence"></span>
      <div class="track-actions">
        <button class="btn auth"     title="AUTHORIZE engagement" data-action="AUTHORIZE">AUTH</button>
        <button class="btn hold"     title="HOLD — block engagement" data-action="HOLD">HOLD</button>
        <button class="btn friendly-btn" title="MARK FRIENDLY" data-action="MARK_FRIENDLY">FRDLY</button>
      </div>
    `;
    row.querySelectorAll<HTMLButtonElement>("[data-action]").forEach(btn => {
      btn.addEventListener("click", () => {
        const action = btn.dataset["action"]!;
        this.ws.send({ type: action as "AUTHORIZE" | "HOLD" | "MARK_FRIENDLY", track_id: trackId });
      });
    });
    return row;
  }

  private _bindWeaponsHold(): void {
    this.weaponsHoldBtn.addEventListener("click", () => {
      const currentlyHeld = this.weaponsHoldBtn.classList.contains("hold-active");
      this.ws.send({ type: "WEAPONS_HOLD", active: !currentlyHeld });
    });
  }
}
