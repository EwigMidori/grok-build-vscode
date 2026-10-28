import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { buildRequest, ResponsesClient, type Item } from "../adapters/deepseek/responses.mts";
import { DeepSeekSession } from "../adapters/deepseek/session.mts";

const dirs: string[] = [];
afterEach(async () => { await Promise.all(dirs.splice(0).map(d => rm(d, { recursive: true, force: true }))); });
const message = (text: string) => ({ type: "message", role: "assistant", content: [{ type: "output_text", text }] });
const reasoning = { type: "reasoning", id: "reason", content: [{ type: "reasoning_text", text: "Consider the file." }] };
const call = (id: string, name: string, args: Item) => ({ type: "function_call", call_id: id, name, arguments: JSON.stringify(args) });
const event = (type: string, extra: Item = {}) => `event: ${type}\r\ndata: ${JSON.stringify({ type, ...extra })}\r\n\r\n`;
const terminal = (output: Item[], status = "completed", extra: Item = {}) => event(`response.${status}`, { response: { output, status, ...extra } });
function stream(source: string, bytewise = false) {
  const bytes = new TextEncoder().encode(source);
  return new Response(new ReadableStream({ start(c) { if (bytewise) for (const byte of bytes) c.enqueue(Uint8Array.of(byte)); else c.enqueue(bytes); c.close(); } }), { headers: { "content-type": "text/event-stream" } });
}
function responses(sources: string[]) {
  const requests: Item[] = [];
  const fetcher = vi.fn(async (_url: any, init: any) => { requests.push(JSON.parse(init.body)); const source = sources.shift(); if (source === undefined) throw new Error("Unexpected fetch"); return stream(source); });
  return { api: new ResponsesClient("http://127.0.0.1:1234/responses", "ephemeral-secret", fetcher as any), requests, fetcher };
}
async function setup(sources: string[], handler?: (method: string, p: any, options?: any) => Promise<any>) {
  const dir = await mkdtemp(join(tmpdir(), "deepseek-adapter-")); dirs.push(dir);
  const transport = responses(sources);
  const client = { request: vi.fn(handler ?? (async (method: string) => {
    if (method === "session/request_permission") return { outcome: { outcome: "selected", optionId: "allow" } };
    if (method === "fs/read_text_file") return { content: "file content" };
    if (method === "terminal/create") return { terminalId: "terminal" };
    if (method === "terminal/wait_for_exit") return { exitCode: 0 };
    if (method === "terminal/output") return { output: "command output", truncated: false };
    return {};
  })), notify: vi.fn(async () => {}) };
  const session = new DeepSeekSession(client as any, transport.api, dir);
  session.initialize({ fs: { readTextFile: true, writeTextFile: true }, terminal: true });
  await mkdir(join(dir, "workspace", "src"), { recursive: true });
  const cwd = await realpath(join(dir, "workspace"));
  await writeFile(join(cwd, "src", "a.ts"), "fixture");
  await writeFile(join(cwd, "a"), "fixture");
  const { sessionId: id } = await session.newSession(cwd);
  const saved = async () => JSON.parse(await readFile(join(dir, `${id}.json`), "utf8"));
  return { ...transport, session, client, id, dir, cwd, saved };
}
const prompt = [{ type: "text" as const, text: "Help with the code" }];

describe("DeepSeek Responses transport", () => {
  it("builds stateless Responses requests with plain reasoning and no server history handles", () => {
    const history = [reasoning, call("c1", "read_file", { path: "a" }), { type: "function_call_output", call_id: "c1", output: "data" }];
    expect(buildRequest("deepseek-flash", "max", history)).toMatchObject({ model: "deepseek-flash", reasoning: { effort: "max" }, input: history, stream: true });
    expect(buildRequest("deepseek-flash", "max", history)).not.toHaveProperty("previous_response_id");
    expect(buildRequest("deepseek-flash", "max", history)).not.toHaveProperty("conversation");
    expect(() => buildRequest("bad", "high", [])).toThrow();
    expect(() => buildRequest("deepseek-flash", "medium", [])).toThrow();
  });
  it("handles fragmented UTF-8 and CRLF, reasoning deltas, and final text without duplication", async () => {
    const output: Item[] = [], deltas: any[] = [];
    const source = event("response.output_item.added", { output_index: 0, item: { type: "reasoning", content: [] } })
      + event("response.reasoning_text.delta", { output_index: 0, content_index: 0, delta: "考え" })
      + event("response.output_text.delta", { output_index: 1, content_index: 0, delta: "Hé" })
      + terminal([{ type: "reasoning", content: [{ type: "reasoning_text", text: "考え" }] }, message("Héllo")]);
    const api = new ResponsesClient("http://[::1]:12/v1/responses", "secret", vi.fn(async () => stream(source, true)) as any);
    const result = await api.run(buildRequest("deepseek-flash", "low", []), output, new AbortController().signal, async (t, r) => { deltas.push([t, r]); });
    expect(result.status).toBe("completed");
    expect(deltas).toEqual([["考え", true], ["Hé", false], ["llo", false]]);
    expect(output[0].content[0]).toEqual({ type: "reasoning_text", text: "考え" });
  });
  it("supports multiline data and comment frames", async () => {
    const source = ': ping\r\n\r\nevent: response.completed\r\ndata: {"response":\r\ndata: {"output":[]}}\r\n\r\n';
    const { api } = responses([source]);
    expect((await api.run(buildRequest("deepseek-flash", "none", []), [], new AbortController().signal, async () => {})).status).toBe("completed");
  });
  it.each(["failed", "incomplete"])("handles terminal response.%s", async status => {
    const { api } = responses([terminal([message("partial")], status, { error: { message: "secret" }, incomplete_details: { reason: "max_output_tokens" } })]);
    const output: Item[] = [];
    const result = api.run(buildRequest("deepseek-flash", "high", []), output, new AbortController().signal, async () => {});
    if (status === "failed") await expect(result).rejects.toThrow("DeepSeek response failed");
    else expect((await result).status).toBe("incomplete");
    expect(output).toEqual([message("partial")]);
  });
  it.each(["", 'data: [DONE]\n\n', 'data: {broken}\n\n'])("rejects premature or invalid EOF (%s)", async source => {
    const { api } = responses([source]);
    await expect(api.run(buildRequest("deepseek-flash", "none", []), [], new AbortController().signal, async () => {})).rejects.toThrow(/DeepSeek/);
  });
  it("sanitizes network errors and HTTP bodies and sends only relay authorization", async () => {
    const fetcher = vi.fn(async () => new Response("sensitive-body", { status: 401 }));
    const api = new ResponsesClient("http://127.0.0.1:12/responses", "secret", fetcher as any);
    await expect(api.run(buildRequest("deepseek-flash", "none", []), [], new AbortController().signal, async () => {})).rejects.toThrow("DeepSeek relay HTTP 401");
    expect((fetcher.mock.calls as any)[0][1].headers.Authorization).toBe("Bearer secret");
    const broken = new ResponsesClient("http://127.0.0.1:12/responses", "secret", (async () => { throw new Error("secret http://private"); }) as any);
    await expect(broken.run(buildRequest("deepseek-flash", "none", []), [], new AbortController().signal, async () => {})).rejects.toThrow("DeepSeek response transport or stream error");
  });
  it.each(["https://api.deepseek.com/responses", "http://localhost/responses", "http://127.0.0.1/chat/completions", "http://user:pass@127.0.0.1/responses", "http://127.0.0.1/responses?key=secret"])("rejects unsafe relay %s", url => {
    expect(() => new ResponsesClient(url, "secret")).toThrow();
  });
  it("aborts a stalled stream and cancels its reader", async () => {
    const cancelled = vi.fn();
    const api = new ResponsesClient("http://127.0.0.1/responses", "secret", (async () => new Response(new ReadableStream({ cancel: cancelled }))) as any);
    const c = new AbortController();
    const running = api.run(buildRequest("deepseek-flash", "high", []), [], c.signal, async () => {});
    await Promise.resolve(); await Promise.resolve(); c.abort();
    await expect(running).rejects.toThrow("cancelled"); expect(cancelled).toHaveBeenCalled();
  });
  it("sanitizes asynchronous stream read failures", async () => {
    const api = new ResponsesClient("http://127.0.0.1/responses", "secret", (async () => new Response(new ReadableStream({ pull(c) { c.error(new Error("secret-from-network")); } }))) as any);
    await expect(api.run(buildRequest("deepseek-flash", "high", []), [], new AbortController().signal, async () => {})).rejects.toThrow("DeepSeek response transport or stream error");
  });
  it("reconstructs function argument deltas and respects final canonical ordering", async () => {
    const tool = call("c", "read_file", { path: "a" });
    const source = event("response.output_item.added", { output_index: 0, item: { ...tool, arguments: "" } })
      + event("response.function_call_arguments.delta", { output_index: 0, delta: tool.arguments })
      + event("response.output_item.done", { output_index: 0, item: tool }) + terminal([tool]);
    const { api } = responses([source]); const output: Item[] = [];
    await api.run(buildRequest("deepseek-flash", "high", []), output, new AbortController().signal, async () => {});
    expect(output).toEqual([tool]);
  });
});

describe("DeepSeek ACP sessions", () => {
  it.each(["absolute", "parent", "sibling-prefix"])("rejects %s outside-workspace file paths without ACP access", async kind => {
    const sources: string[] = [];
    const s = await setup(sources);
    const outside = kind === "sibling-prefix" ? `${s.cwd}-other` : join(s.dir, "outside");
    await mkdir(outside); await writeFile(join(outside, "auth.json"), "private credentials");
    const path = kind === "parent" ? "../outside/auth.json" : join(outside, "auth.json");
    sources.push(terminal([call("read-outside", "read_file", { path }), call("write-outside", "write_file", { path, content: "bad" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request).not.toHaveBeenCalled();
    expect(s.requests[1].input.filter((i: Item) => i.type === "function_call_output").map((i: Item) => i.output)).toEqual(["Tool failed or permission denied.", "Tool failed or permission denied."]);
    expect(await readFile(join(outside, "auth.json"), "utf8")).toBe("private credentials");
  });
  it.each(["auth.json", "new/nested/file.txt"])("rejects symlink escape for existing or missing targets (%s)", async name => {
    const sources: string[] = [];
    const s = await setup(sources);
    const outside = join(s.dir, "outside");
    await mkdir(outside); await writeFile(join(outside, "auth.json"), "private credentials");
    await symlink(outside, join(s.cwd, "escape"), process.platform === "win32" ? "junction" : "dir");
    const path = join("escape", name);
    sources.push(terminal([call("read-link", "read_file", { path }), call("write-link", "write_file", { path, content: "bad" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request).not.toHaveBeenCalled();
    expect(JSON.stringify(s.requests)).not.toContain("private credentials");
    expect(await readFile(join(outside, "auth.json"), "utf8")).toBe("private credentials");
  });
  it("allows internal symlinks and passes the canonical in-workspace path to ACP", async () => {
    const sources: string[] = [];
    const s = await setup(sources);
    await symlink(join(s.cwd, "src"), join(s.cwd, "linked"), process.platform === "win32" ? "junction" : "dir");
    sources.push(terminal([call("read-internal", "read_file", { path: "linked/a.ts" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request).toHaveBeenCalledWith("fs/read_text_file", { sessionId: s.id, path: join(s.cwd, "src", "a.ts") }, expect.anything());
  });
  it("allows an absolute in-workspace file and approved creation through missing directories", async () => {
    const sources: string[] = [];
    const s = await setup(sources);
    sources.push(terminal([call("read-absolute", "read_file", { path: join(s.cwd, "a") }), call("new-write", "write_file", { path: "new/nested/file.txt", content: "new" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request.mock.calls.map(c => c[0])).toEqual(["fs/read_text_file", "session/request_permission", "fs/write_text_file"]);
    expect(s.client.request.mock.calls.at(-1)?.[1]).toEqual({ sessionId: s.id, path: join(s.cwd, "new", "nested", "file.txt"), content: "new" });
  });
  it("rejects a write symlink changed to an outside target while awaiting approval", async () => {
    const sources: string[] = [];
    let cwd = "", outside = "";
    const s = await setup(sources, async method => {
      if (method === "session/request_permission") {
        await rm(join(cwd, "linked"));
        await symlink(outside, join(cwd, "linked"), process.platform === "win32" ? "junction" : "dir");
        return { outcome: { outcome: "selected", optionId: "allow" } };
      }
      throw new Error("Unexpected file access");
    });
    cwd = s.cwd; outside = join(s.dir, "outside");
    await mkdir(outside);
    await symlink(join(cwd, "src"), join(cwd, "linked"), process.platform === "win32" ? "junction" : "dir");
    sources.push(terminal([call("write-swapped", "write_file", { path: "linked/a.ts", content: "bad" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request.mock.calls.map(c => c[0])).toEqual(["session/request_permission"]);
  });
  it("rejects dangling write symlinks rather than treating them as creatable directories", async () => {
    const sources: string[] = [];
    const s = await setup(sources);
    await symlink(join(s.dir, "missing-outside"), join(s.cwd, "dangling"), process.platform === "win32" ? "junction" : "dir");
    sources.push(terminal([call("write-dangling", "write_file", { path: "dangling/new.txt", content: "bad" })]), terminal([]));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request).not.toHaveBeenCalled();
  });
  it("rejects directory reads before ACP access", async () => {
    const s = await setup([terminal([call("read-dir", "read_file", { path: "src" })]), terminal([])]);
    await s.session.prompt(s.id, prompt);
    expect(s.client.request).not.toHaveBeenCalled();
  });
  it("records malformed ACP file responses as tool failures without poisoning persisted history", async () => {
    const s = await setup([terminal([call("read-invalid", "read_file", { path: "a" })]), terminal([])], async () => ({}));
    await s.session.prompt(s.id, prompt);
    expect(s.requests[1].input.at(-1).output).toBe("Tool failed or permission denied.");
    await s.session.loadSession(s.id, s.cwd);
  });
  it("replays full canonical output and matching results across turns and cold loads", async () => {
    const tool = call("read-1", "read_file", { path: "src/a.ts" });
    const s = await setup([terminal([reasoning, tool]), terminal([message("Done")]), terminal([message("Again")])]);
    expect((await s.session.prompt(s.id, prompt)).stopReason).toBe("end_turn");
    expect(s.requests[1].input).toEqual([{ type: "message", role: "user", content: [{ type: "input_text", text: prompt[0].text }] }, reasoning, tool, { type: "function_call_output", call_id: "read-1", output: "file content" }]);
    const cold = new DeepSeekSession(s.client as any, s.api, s.dir);
    cold.initialize({ fs: { readTextFile: true } });
    await cold.loadSession(s.id, s.cwd);
    await cold.prompt(s.id, prompt);
    expect(s.requests[2].input.slice(0, -1)).toEqual((await s.saved()).history.slice(0, -2));
    expect(s.requests[2]).not.toHaveProperty("previous_response_id");
    expect(s.client.request).toHaveBeenCalledWith("fs/read_text_file", expect.objectContaining({ path: resolve(s.cwd, "src/a.ts") }), expect.anything());
  });
  it("keeps call ordering and asks before writes and executable/argv commands without env", async () => {
    const calls = [call("w1", "write_file", { path: "a.ts", content: "new" }), call("r1", "run_command", { command: "node", args: ["-e", "process.exit(0)"] })];
    const s = await setup([terminal([reasoning, ...calls]), terminal([message("Finished")])]);
    await s.session.prompt(s.id, prompt);
    const methods = s.client.request.mock.calls.map(c => c[0]);
    expect(methods).toEqual(["session/request_permission", "fs/write_text_file", "session/request_permission", "terminal/create", "terminal/wait_for_exit", "terminal/output", "terminal/release"]);
    const create = s.client.request.mock.calls.find(c => c[0] === "terminal/create")![1];
    expect(create).toMatchObject({ command: "node", args: ["-e", "process.exit(0)"], cwd: s.cwd });
    expect(create).not.toHaveProperty("env");
    expect(JSON.stringify(s.requests)).not.toContain("ephemeral-secret");
    expect((await s.saved()).history.slice(1, 4)).toEqual([reasoning, ...calls]);
  });
  it.each(["reject", "cancelled", "unknown"])("does not write after permission %s", async outcome => {
    const s = await setup([terminal([call("w", "write_file", { path: "a", content: "bad" })]), terminal([message("Denied")])], async () => ({ outcome: outcome === "cancelled" ? { outcome } : { outcome: "selected", optionId: outcome } }));
    await s.session.prompt(s.id, prompt);
    expect(s.client.request.mock.calls.map(c => c[0])).toEqual(["session/request_permission"]);
    expect(s.requests[1].input.at(-1).output).toMatch(/failed or permission denied/);
  });
  it("cancels before permission resolves without an unauthorized write and pairs remaining calls", async () => {
    let release!: (value: any) => void;
    let requested!: () => void;
    const ready = new Promise<void>(r => { requested = r; });
    const s = await setup([terminal([call("w", "write_file", { path: "a", content: "bad" }), call("next", "write_file", { path: "b", content: "bad" })])], async () => { requested(); return new Promise(r => { release = r; }); });
    const running = s.session.prompt(s.id, prompt);
    await ready; s.session.cancel(s.id); release({ outcome: { outcome: "selected", optionId: "allow" } });
    expect((await running).stopReason).toBe("cancelled");
    expect(s.client.request.mock.calls.map(c => c[0])).toEqual(["session/request_permission"]);
    expect((await s.saved()).history.filter((i: Item) => i.type === "function_call_output")).toHaveLength(2);
  });
  it("settles Stop while a permission remains unanswered and sends SDK cancellation", async () => {
    let requested!: () => void;
    const ready = new Promise<void>(r => { requested = r; });
    const s = await setup([terminal([call("w", "write_file", { path: "a", content: "bad" })])], async () => { requested(); return new Promise(() => {}); });
    const running = s.session.prompt(s.id, prompt); await ready;
    await expect(s.session.prompt(s.id, prompt)).rejects.toThrow(/busy/);
    await expect(s.session.setModel(s.id, "deepseek-v4-pro")).rejects.toThrow(/busy/);
    s.session.cancel(s.id);
    expect((await running).stopReason).toBe("cancelled");
    expect(s.client.request.mock.calls[0][2].cancellationSignal.aborted).toBe(true);
    expect(s.client.request).toHaveBeenCalledTimes(1);
  });
  it("releases an active terminal when its wait ignores cancellation", async () => {
    let waiting!: () => void;
    const ready = new Promise<void>(r => { waiting = r; });
    const s = await setup([terminal([call("r", "run_command", { command: "node", args: [] })])], async method => {
      if (method === "session/request_permission") return { outcome: { outcome: "selected", optionId: "allow" } };
      if (method === "terminal/create") return { terminalId: "t" };
      if (method === "terminal/wait_for_exit") { waiting(); return new Promise(() => {}); }
      return {};
    });
    const running = s.session.prompt(s.id, prompt); await ready; s.session.cancel(s.id);
    expect((await running).stopReason).toBe("cancelled");
    expect(s.client.request.mock.calls.at(-1)).toEqual(["terminal/release", { sessionId: s.id, terminalId: "t" }, expect.objectContaining({ cancellationSignal: expect.any(AbortSignal) })]);
  });
  it("reclaims a terminal whose creation finishes after cancellation", async () => {
    let started!: () => void, created!: (value: any) => void;
    const ready = new Promise<void>(r => { started = r; });
    const s = await setup([terminal([call("r", "run_command", { command: "node", args: [] })])], async method => {
      if (method === "session/request_permission") return { outcome: { outcome: "selected", optionId: "allow" } };
      if (method === "terminal/create") { started(); return new Promise(r => { created = r; }); }
      return {};
    });
    const running = s.session.prompt(s.id, prompt); await ready; s.session.cancel(s.id);
    expect((await running).stopReason).toBe("cancelled");
    created({ terminalId: "late" });
    await vi.waitFor(() => expect(s.client.request.mock.calls.at(-1)).toEqual(["terminal/release", { sessionId: s.id, terminalId: "late" }, expect.objectContaining({ cancellationSignal: expect.any(AbortSignal) })]));
  });
  it("always releases a terminal after host failure and records a paired failure", async () => {
    const s = await setup([terminal([call("r", "run_command", { command: "node", args: [] })]), terminal([])], async method => {
      if (method === "session/request_permission") return { outcome: { outcome: "selected", optionId: "allow" } };
      if (method === "terminal/create") return { terminalId: "t" };
      if (method === "terminal/wait_for_exit") throw new Error("host-secret");
      return {};
    });
    await s.session.prompt(s.id, prompt);
    expect(s.client.request.mock.calls.at(-1)?.[0]).toBe("terminal/release");
    expect(s.requests[1].input.at(-1).output).not.toContain("host-secret");
  });
  it("pairs calls after interrupted SSE and recovers interrupted persisted histories", async () => {
    const tool = call("interrupted", "write_file", { path: "a", content: "bad" });
    const s = await setup([event("response.output_item.done", { output_index: 0, item: tool })]);
    await expect(s.session.prompt(s.id, prompt)).rejects.toThrow(/terminal response/);
    expect(s.client.request).not.toHaveBeenCalled();
    const saved = await s.saved();
    expect(saved.history.at(-1)).toMatchObject({ type: "function_call_output", call_id: "interrupted" });
    saved.history.pop(); await writeFile(join(s.dir, `${s.id}.json`), JSON.stringify(saved));
    await s.session.loadSession(s.id, s.cwd);
    expect((await s.saved()).history.at(-1).call_id).toBe("interrupted");
  });
  it("does not execute calls in incomplete responses and returns usage metadata", async () => {
    const s = await setup([terminal([call("w", "write_file", { path: "a", content: "bad" })], "incomplete", { usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 }, incomplete_details: { reason: "max_output_tokens" } })]);
    const result = await s.session.prompt(s.id, prompt);
    expect(result).toMatchObject({ stopReason: "max_tokens", _meta: { deepseek: { usage: { total_tokens: 3 } } } });
    expect(s.client.request).not.toHaveBeenCalled();
  });
  it("retains failed-response canonical output without executing calls", async () => {
    const s = await setup([terminal([reasoning, call("w", "write_file", { path: "a", content: "bad" })], "failed", { error: { message: "body-secret" } })]);
    await expect(s.session.prompt(s.id, prompt)).rejects.toThrow("DeepSeek response failed");
    expect((await s.saved()).history.slice(1, 3)).toEqual([reasoning, call("w", "write_file", { path: "a", content: "bad" })]);
    expect((await s.saved()).history.at(-1).call_id).toBe("w");
    expect(s.client.request).not.toHaveBeenCalled();
  });
  it("rejects duplicate tool IDs before side effects or persisting malformed history", async () => {
    const tool = call("duplicate", "write_file", { path: "a", content: "bad" });
    const s = await setup([terminal([tool, tool])]);
    await expect(s.session.prompt(s.id, prompt)).rejects.toThrow(/invalid response items/);
    expect(s.client.request).not.toHaveBeenCalled();
    expect((await s.saved()).history).toHaveLength(1);
    await s.session.loadSession(s.id, s.cwd);
  });
  it("reports summed ACP usage across model/tool round trips", async () => {
    const usage = { input_tokens: 10, output_tokens: 5, total_tokens: 15, input_tokens_details: { cached_tokens: 2 }, output_tokens_details: { reasoning_tokens: 3 } };
    const s = await setup([terminal([call("read", "read_file", { path: "a" })], "completed", { usage }), terminal([message("Done")], "completed", { usage })]);
    expect((await s.session.prompt(s.id, prompt)).usage).toEqual({ inputTokens: 20, outputTokens: 10, totalTokens: 30, cachedReadTokens: 4, thoughtTokens: 6 });
    expect(s.client.notify.mock.calls.some((c: any) => c[1].update.sessionUpdate === "session_info_update")).toBe(true);
  });
  it("advertises only agent mode, persists model/effort changes and rejects unsupported modes", async () => {
    const s = await setup([terminal([])]);
    expect((await s.session.loadSession(s.id, s.cwd)).modes.availableModes.map(m => m.id)).toEqual(["agent"]);
    await expect(s.session.setMode(s.id, "yolo")).rejects.toThrow();
    await expect(s.session.setMode(s.id, "plan")).rejects.toThrow();
    await s.session.setModel(s.id, "deepseek-v4-pro"); await s.session.setReasoningEffort(s.id, "max");
    await s.session.prompt(s.id, prompt);
    expect(s.requests[0]).toMatchObject({ model: "deepseek-v4-pro", reasoning: { effort: "max" } });
    expect(await s.saved()).toMatchObject({ model: "deepseek-v4-pro", effort: "max" });
  });
  it("accepts flash images, rejects pro images and switching image histories to pro", async () => {
    const s = await setup([terminal([])]);
    const image = [{ type: "image" as const, mimeType: "image/png", data: "aGVsbG8=" }];
    await s.session.prompt(s.id, image);
    expect(s.requests[0].input[0].content[0]).toEqual({ type: "input_image", image_url: "data:image/png;base64,aGVsbG8=" });
    await expect(s.session.setModel(s.id, "deepseek-v4-pro")).rejects.toThrow(/images/);
    const fresh = await s.session.newSession(s.cwd); await s.session.setModel(fresh.sessionId, "deepseek-v4-pro");
    await expect(s.session.prompt(fresh.sessionId, image)).rejects.toThrow(/images/);
  });
  it("lists only valid sessions, filters workspace and refuses traversal or corrupt content", async () => {
    const s = await setup([]);
    await writeFile(join(s.dir, "garbage.json"), "{}");
    const corrupt = "12345678-1234-4234-8234-123456789012";
    await writeFile(join(s.dir, `${corrupt}.json`), JSON.stringify({ ...(await s.saved()), id: corrupt, history: [{ type: "function_call_output", call_id: "missing", output: "bad" }] }));
    expect((await s.session.listSessions()).sessions.map(i => i.sessionId)).toEqual([s.id]);
    expect((await s.session.listSessions(resolve("elsewhere"))).sessions).toEqual([]);
    await expect(s.session.loadSession("../secret", s.cwd)).rejects.toThrow(/ID/);
    await expect(s.session.loadSession(corrupt, s.cwd)).rejects.toThrow(/history/);
    await expect(s.session.loadSession(s.id, resolve("elsewhere"))).rejects.toThrow(/workspace/);
  });
  it("omits unavailable host tools", async () => {
    const s = await setup([terminal([])]); s.session.initialize({});
    await s.session.prompt(s.id, prompt); expect(s.requests[0].tools).toEqual([]);
  });
  it("surfaces persistence failure without admitting a network request", async () => {
    const s = await setup([terminal([])]);
    await rm(s.dir, { recursive: true }); await writeFile(s.dir, "not a directory");
    await expect(s.session.prompt(s.id, prompt)).rejects.toThrow(/persist/);
    expect(s.fetcher).not.toHaveBeenCalled();
  });
});
