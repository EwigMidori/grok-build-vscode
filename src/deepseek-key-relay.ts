import { randomBytes } from "node:crypto";
import * as http from "node:http";
import * as https from "node:https";
import type { AddressInfo } from "node:net";
import type { AgyUpstreamRequest } from "./agy-key-relay";

export const DEEPSEEK_RESPONSES_UPSTREAM = "https://api.deepseek.com/responses";
export interface DeepSeekRelayLease {
  responsesUrl: string;
  token: string;
  release(): void;
}

/** One process-scoped capability, not a general proxy. Pipes preserve SSE backpressure. */
export async function createDeepSeekRelay(key: string,
  upstream: AgyUpstreamRequest = (url, options) => https.request(url, options),
): Promise<DeepSeekRelayLease> {
  const token = randomBytes(32).toString("base64url");
  const pending = new Set<http.ClientRequest>();
  const server = http.createServer({ requestTimeout: 0 }, (req, res) => {
    if (req.url !== "/responses" || req.method !== "POST" || req.headers.authorization !== `Bearer ${token}`) {
      res.writeHead(403); res.end("Forbidden"); req.resume(); return;
    }
    let outgoing: http.ClientRequest;
    const fail = () => {
      if (!res.headersSent) { res.writeHead(502); res.end("DeepSeek upstream request failed"); }
      else res.destroy();
    };
    try {
      outgoing = upstream(new URL(DEEPSEEK_RESPONSES_UPSTREAM), { method: "POST", headers: {
        authorization: `Bearer ${key}`, "content-type": "application/json", accept: "text/event-stream, application/json",
      } });
    } catch { fail(); req.resume(); return; }
    pending.add(outgoing);
    outgoing.once("close", () => pending.delete(outgoing));
    outgoing.on("error", fail);
    req.on("error", () => outgoing.destroy());
    req.on("aborted", () => outgoing.destroy());
    res.on("close", () => outgoing.destroy());
    outgoing.on("response", (incoming) => {
      incoming.on("error", fail);
      incoming.on("aborted", () => { fail(); outgoing.destroy(); });
      res.writeHead(incoming.statusCode ?? 502, { "content-type": incoming.headers["content-type"] ?? "application/json" });
      incoming.pipe(res);
    });
    req.pipe(outgoing);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => { server.off("error", reject); resolve(); });
  });
  let released = false;
  return { responsesUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}/responses`, token,
    release() {
      if (released) return;
      released = true;
      for (const request of pending) request.destroy();
      pending.clear();
      server.close(); server.closeAllConnections();
    } };
}
