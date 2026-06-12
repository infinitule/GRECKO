/**
 * Replay scrubber — load a JSONL event log and play it back at configurable
 * speed without re-running physics.
 *
 * Replay determinism: the hash of the replayed events must match the hash
 * of the original log.
 */
import type { SimState } from "./types.js";

interface ReplayEvent {
  t: number;
  type: string;
  [key: string]: unknown;
}

export class ReplayPlayer {
  private events: ReplayEvent[] = [];
  private frameIdx = 0;
  private playing = false;
  private speed = 1.0;
  private _raf: number | null = null;
  private _onFrame: ((state: Partial<SimState>) => void) | null = null;
  private _lastWall = 0;
  private _replaySim = 0;

  onFrame(cb: (state: Partial<SimState>) => void): void {
    this._onFrame = cb;
  }

  async loadFile(file: File): Promise<void> {
    const text = await file.text();
    this.events = text
      .split("\n")
      .filter(Boolean)
      .map(line => {
        try { return JSON.parse(line) as ReplayEvent; }
        catch { return null; }
      })
      .filter(Boolean) as ReplayEvent[];
    this.frameIdx = 0;
    this._replaySim = this.events[0]?.t ?? 0;
  }

  play(speed = 1.0): void {
    this.speed = speed;
    if (this.playing) return;
    this.playing = true;
    this._lastWall = performance.now();
    this._tick();
  }

  pause(): void {
    this.playing = false;
    if (this._raf !== null) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  seekTo(fraction: number): void {
    if (this.events.length === 0) return;
    const maxT = this.events[this.events.length - 1]?.t ?? 0;
    const target = fraction * maxT;
    this.frameIdx = this.events.findIndex(e => e.t >= target);
    if (this.frameIdx < 0) this.frameIdx = this.events.length - 1;
    this._replaySim = this.events[this.frameIdx]?.t ?? 0;
  }

  get duration(): number {
    return this.events[this.events.length - 1]?.t ?? 0;
  }

  get currentTime(): number {
    return this._replaySim;
  }

  private _tick(): void {
    if (!this.playing) return;
    const now = performance.now();
    const elapsed = (now - this._lastWall) / 1000.0;  // seconds
    this._lastWall = now;
    this._replaySim += elapsed * this.speed;

    // Emit all events up to current replay time
    while (
      this.frameIdx < this.events.length &&
      (this.events[this.frameIdx]?.t ?? Infinity) <= this._replaySim
    ) {
      const ev = this.events[this.frameIdx++];
      if (ev && this._onFrame) {
        // Wrap event as a minimal SimState-like object for renderer compatibility
        this._onFrame({ t: ev.t } as Partial<SimState>);
      }
    }

    if (this.frameIdx >= this.events.length) {
      this.playing = false;
      return;
    }

    this._raf = requestAnimationFrame(() => this._tick());
  }
}
