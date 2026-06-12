/**
 * WebSocket client with auto-reconnect and typed message dispatch.
 *
 * Usage:
 *   const ws = new C2WebSocket("ws://127.0.0.1:8765");
 *   ws.onState(state => { ... });
 *   ws.send({ type: "AUTHORIZE", track_id: "T0042" });
 */
import type { C2Command, ServerMessage, SimState } from "./types.js";

type StateHandler = (state: SimState) => void;
type RawHandler = (msg: ServerMessage) => void;

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];

export class C2WebSocket {
  private url: string;
  private _ws: WebSocket | null = null;
  private _stateHandlers: StateHandler[] = [];
  private _rawHandlers: RawHandler[] = [];
  private _reconnectAttempt = 0;
  private _intentionallyClosed = false;

  constructor(url: string = "ws://127.0.0.1:8765") {
    this.url = url;
    this._connect();
  }

  onState(handler: StateHandler): () => void {
    this._stateHandlers.push(handler);
    return () => {
      this._stateHandlers = this._stateHandlers.filter(h => h !== handler);
    };
  }

  onMessage(handler: RawHandler): () => void {
    this._rawHandlers.push(handler);
    return () => {
      this._rawHandlers = this._rawHandlers.filter(h => h !== handler);
    };
  }

  send(cmd: C2Command): void {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(cmd));
    }
  }

  get isConnected(): boolean {
    return this._ws?.readyState === WebSocket.OPEN;
  }

  close(): void {
    this._intentionallyClosed = true;
    this._ws?.close();
  }

  // ------------------------------------------------------------------ //

  private _connect(): void {
    if (this._intentionallyClosed) return;
    this._ws = new WebSocket(this.url);

    this._ws.addEventListener("open", () => {
      this._reconnectAttempt = 0;
      document.getElementById("connection-indicator")?.classList.replace(
        "disconnected", "connected"
      );
      document.getElementById("connection-indicator")?.classList.add("connected");
    });

    this._ws.addEventListener("message", (ev: MessageEvent<string>) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(ev.data) as ServerMessage;
      } catch {
        return;
      }
      this._rawHandlers.forEach(h => h(msg));
      if (!("type" in msg) || msg.type === undefined) {
        // It's a SimState broadcast
        this._stateHandlers.forEach(h => h(msg as SimState));
      }
    });

    this._ws.addEventListener("close", () => {
      const el = document.getElementById("connection-indicator");
      el?.classList.remove("connected");
      el?.classList.add("disconnected");
      if (!this._intentionallyClosed) {
        const delay = RECONNECT_DELAYS_MS[
          Math.min(this._reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)
        ]!;
        this._reconnectAttempt++;
        setTimeout(() => this._connect(), delay);
      }
    });
  }
}
