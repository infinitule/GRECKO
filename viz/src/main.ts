/**
 * AEGISNET C2 console — application entry point.
 *
 * Wires the WebSocket client to all display components:
 *   AirPicture  — canvas-based air picture renderer
 *   HOTLPanel   — per-track AUTHORIZE/HOLD/MARK-FRIENDLY and WEAPONS HOLD
 *   AuditTrail  — scrollable audit event log
 *   ROEEditor   — lambda slider
 */
import { C2WebSocket } from "./ws.js";
import { AirPicture } from "./air_picture.js";
import { HOTLPanel } from "./hotl_panel.js";
import { AuditTrail } from "./audit_trail.js";
import { ROEEditor } from "./roe_editor.js";
import type { SimState } from "./types.js";

const WS_URL = (window as Window & { AEGISNET_WS_URL?: string }).AEGISNET_WS_URL
  ?? "ws://127.0.0.1:8765";

function main(): void {
  const ws = new C2WebSocket(WS_URL);

  const airPicture = new AirPicture(
    document.getElementById("air-picture") as HTMLCanvasElement
  );
  const hotlPanel = new HOTLPanel(ws);
  const auditTrail = new AuditTrail();
  const roeEditor = new ROEEditor(ws);

  const simTimeEl = document.getElementById("sim-time")!;
  const trackCountEl = document.getElementById("track-count")!;
  const assetHpEl = document.getElementById("asset-hp")!;

  ws.onState((state: SimState) => {
    // Header
    simTimeEl.textContent = `T+${state.t.toFixed(2)}s`;
    trackCountEl.textContent = `${state.tracks.filter(t => t.status !== "tentative").length} tracks`;
    assetHpEl.textContent = `HP:${state.asset.hp.toFixed(0)}`;

    // Components
    airPicture.render(state);
    hotlPanel.render(state);
    roeEditor.update(state);
    auditTrail.update(state.audit_trail);
  });
}

main();
