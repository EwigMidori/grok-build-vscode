import * as http from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createDeepSeekRelay, DEEPSEEK_RESPONSES_UPSTREAM, type DeepSeekRelayLease } from "../src/deepseek-key-relay";

let fake: http.Server | undefined;
let relay: DeepSeekRelayLease | undefined;
afterEach(async () => {
  relay?.release();
  if (fake) {
    fake.closeAllConnections();
    await new Promise<void>(resolve => fake!.close(() => resolve()));
  }
  fake = undefined; relay = undefined;
});

async function setup(handler: http.RequestListener) {
  fake = http.createServer(handler);
  await new Promise<void>(resolve => fake!.listen(0, "127.0.0.1", resolve));
  const port = (fake.address() as AddressInfo).port;
  const targets: string[] = [];
  relay = await createDeepSeekRelay("saved-key", (url, options) => {
    targets.push(String(url));
    return http.request({ host: "127.0.0.1", port, path: url.pathname, ...options });
  });
  return targets;
}

function send(url = relay!.responsesUrl, authorization = `Bearer ${relay!.token}`, method = "POST") {
  return new Promise<{ status: number; body: string }>((resolve, reject) => {
    const bodyText = '{"model":"deepseek-flash","stream":true}';
    const request = http.request(url, { method, headers: { authorization, "x-api-key": "must-not-forward", "content-length": Buffer.byteLength(bodyText) } }, res => {
      let body = "";
      res.on("data", chunk => { body += chunk; });
      res.on("error", reject);
      res.on("end", () => resolve({ status: res.statusCode!, body }));
    });
    request.on("error", reject);
    request.end(bodyText);
  });
}

describe("DeepSeek fixed-target credential relay", () => {
  it("forwards POST only to Responses, substitutes the key, and streams SSE unchanged", async () => {
    let headers: http.IncomingHttpHeaders = {};
    let body = "";
    const targets = await setup((req, res) => {
      headers = req.headers;
      req.on("data", chunk => { body += chunk; });
      req.on("end", () => { res.writeHead(200, { "content-type": "text/event-stream" }); res.write("data: first\n\n"); setImmediate(() => res.end("data: last\n\n")); });
    });
    expect(await send()).toEqual({ status: 200, body: "data: first\n\ndata: last\n\n" });
    expect(targets).toEqual([DEEPSEEK_RESPONSES_UPSTREAM]);
    expect(headers.authorization).toBe("Bearer saved-key");
    expect(headers["x-api-key"]).toBeUndefined();
    expect(body).toContain('"model":"deepseek-flash"');
  });

  it("rejects missing/wrong capabilities, alternate methods, paths, query and absolute targets", async () => {
    const targets = await setup((_req, res) => res.end());
    const url = relay!.responsesUrl;
    for (const [target, auth, method] of [
      [url, "", "POST"], [url, "Bearer wrong", "POST"], [url, `Bearer ${relay!.token}`, "GET"],
      [url + "?url=https://evil.example", `Bearer ${relay!.token}`, "POST"],
      [url.replace("/responses", "//evil.example/responses"), `Bearer ${relay!.token}`, "POST"],
    ]) expect((await send(target, auth, method)).status).toBe(403);
    expect(targets).toEqual([]);
  });

  it("passes upstream auth failure statuses without copying upstream headers", async () => {
    await setup((_req, res) => { res.writeHead(401, { authorization: "do-not-copy" }); res.end('{"error":"invalid_api_key"}'); });
    expect(await send()).toEqual({ status: 401, body: '{"error":"invalid_api_key"}' });
  });

  it("handles both synchronous and asynchronous upstream failures", async () => {
    relay = await createDeepSeekRelay("key", () => { throw new Error("sensitive"); });
    expect(await send()).toEqual({ status: 502, body: "DeepSeek upstream request failed" });
    relay.release();
    await setup((req) => req.socket.destroy());
    expect(await send()).toEqual({ status: 502, body: "DeepSeek upstream request failed" });
  });

  it.each(["abort", "release"])("cancels the streaming upstream on %s and can release twice", async action => {
    let closed = false;
    await setup((_req, res) => { res.on("close", () => { closed = true; }); res.writeHead(200, { "content-type": "text/event-stream" }); res.write("data: first\n\n"); });
    await new Promise<void>((resolve, reject) => {
      const request = http.request(relay!.responsesUrl, { method: "POST", headers: { authorization: `Bearer ${relay!.token}` } }, res => {
        res.on("error", () => {});
        res.once("data", () => {
          if (action === "abort") request.destroy();
          else relay!.release();
          resolve();
        });
      });
      request.on("error", reject); request.end("{}");
    });
    await vi.waitFor(() => expect(closed).toBe(true));
    relay!.release(); relay!.release();
  });
});
