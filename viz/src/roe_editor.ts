/**
 * ROE (rules-of-engagement) editor.
 *
 * Currently exposes a single λ (lambda_cost) slider that maps to the
 * EconomicMDP cost-exchange knob.  λ=0 → engage everything; λ=1 → only
 * engage when kill_value > cost/asset_value.
 *
 * The slider sends SET_LAMBDA commands to the bridge on change (debounced).
 * It also reflects the current λ from incoming SimState broadcasts.
 */
import type { C2WebSocket } from "./ws.js";
import type { SimState } from "./types.js";

export class ROEEditor {
  private ws: C2WebSocket;
  private slider: HTMLInputElement;
  private valEl: HTMLElement;
  private sendTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(ws: C2WebSocket) {
    this.ws = ws;
    this.slider = document.getElementById("lambda-slider") as HTMLInputElement;
    this.valEl = document.getElementById("lambda-val")!;
    this._bind();
  }

  update(state: SimState): void {
    // Only update the slider if the user isn't currently dragging it
    if (document.activeElement !== this.slider) {
      this.slider.value = String(state.lambda_cost);
      this.valEl.textContent = state.lambda_cost.toFixed(2);
    }
  }

  private _bind(): void {
    this.slider.addEventListener("input", () => {
      const v = parseFloat(this.slider.value);
      this.valEl.textContent = v.toFixed(2);
      if (this.sendTimer !== null) clearTimeout(this.sendTimer);
      this.sendTimer = setTimeout(() => {
        this.ws.send({ type: "SET_LAMBDA", value: v });
      }, 200);
    });
  }
}
