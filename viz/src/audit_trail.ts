/**
 * Scrollable audit trail view.
 *
 * Appends new entries as they arrive.  Keeps at most MAX_ENTRIES visible.
 * Auto-scrolls to newest unless the user has manually scrolled up.
 */
import type { AuditEntry } from "./types.js";

const MAX_ENTRIES = 200;

const EVENT_CLASS: Record<string, string> = {
  AUTHORIZED_ENGAGE:  "authorized",
  WEAPONS_HOLD_ON:    "weapons-hold",
  WEAPONS_HOLD_OFF:   "weapons-hold",
  WEAPONS_HOLD_ACTIVE:"weapons-hold",
  HOLD_PENDING_AUTH:  "pending",
  OPERATOR_HOLD:      "pending",
  OPERATOR_HOLD_BLOCK:"pending",
  MARK_FRIENDLY:      "friendly-mark",
};

export class AuditTrail {
  private list: HTMLElement;
  private seenKeys = new Set<string>();
  private userScrolled = false;

  constructor() {
    this.list = document.getElementById("audit-list")!;
    this.list.addEventListener("scroll", () => {
      const atBottom =
        this.list.scrollTop + this.list.clientHeight >= this.list.scrollHeight - 20;
      this.userScrolled = !atBottom;
    });
  }

  update(entries: AuditEntry[]): void {
    let added = false;
    for (const entry of entries) {
      const key = `${entry.wall_t}_${entry.event}_${entry.track_id}`;
      if (this.seenKeys.has(key)) continue;
      this.seenKeys.add(key);

      const el = document.createElement("div");
      const cls = EVENT_CLASS[entry.event] ?? "";
      el.className = `audit-entry ${cls}`;
      el.innerHTML = `
        <span class="t">T+${entry.sim_t.toFixed(2)}s</span>
        <span class="event">${entry.event}</span>
        <span class="ids">${[entry.interceptor_id, entry.track_id].filter(Boolean).join("→")}</span>
        <span class="detail">${entry.detail ?? ""}</span>
      `;
      this.list.appendChild(el);
      added = true;
    }

    // Trim old entries
    while (this.list.children.length > MAX_ENTRIES) {
      this.list.removeChild(this.list.firstChild!);
    }

    if (added && !this.userScrolled) {
      this.list.scrollTop = this.list.scrollHeight;
    }
  }
}
