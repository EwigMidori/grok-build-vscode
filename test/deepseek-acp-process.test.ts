import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import * as http from "node:http";
import type { AddressInfo } from "node:net";
import { mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createRequire } from "node:module";
import ts from "typescript";
import { AcpClient, type PermissionRequest } from "../src/acp";
import { createDeepSeekRelay, type DeepSeekRelayLease } from "../src/deepseek-key-relay";
import { TerminalManager } from "../src/terminal-manager";

const require = createRequire(import.meta.url);
const root = resolve(__dirname, "..");
const apiKey = "process-test-host-only-key";
let buildDir: string;
let DeepSeekBackend: typeof import("../src/deepseek-backend").DeepSeekBackend;
const cleanup: Array<() => Promise<unknown> | void> = [];

beforeAll(async () => {
  buildDir = await mkdtemp(join(tmpdir(), "deepseek-acp-build-"));
  // Use the shipping compiler settings, never a possibly stale out/ artifact.
  const configFile = ts.readConfigFile(join(root, "tsconfig.deepseek-adapter.json"), ts.sys.readFile);
  if (configFile.error) throw new Error(ts.flattenDiagnosticMessageText(configFile.error.messageText, "\n"));
  const config = ts.parseJsonConfigFileContent(configFile.config, ts.sys, root, { outDir: join(buildDir, "deepseek-adapter") });
  const program = ts.createProgram(config.fileNames, config.options);
  const result = program.emit();
  const errors = [...config.errors, ...ts.getPreEmitDiagnostics(program), ...result.diagnostics];
  expect(errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, "\n"))).toEqual([]);
  expect(result.emitSkipped).toBe(false);
  await symlink(join(root, "node_modules"), join(buildDir, "node_modules"), process.platform === "win32" ? "junction" : "dir");
  // Compile the real backend beside the adapter so its unmodified spawn path works.
  const backend = ts.transpileModule(await readFile(join(root, "src/deepseek-backend.ts"), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  await writeFile(join(buildDir, "deepseek-backend.cjs"), backend.outputText);
  DeepSeekBackend = require(join(buildDir, "deepseek-backend.cjs")).DeepSeekBackend;
});

afterEach(async () => {
  for (const dispose of cleanup.splice(0).reverse()) await dispose();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});
afterAll(async () => { if (buildDir) await rm(buildDir, { recursive: true, force: true }); });

type Item = Record<string, any>;
const message = (text: string) => ({ type: "message", role: "assistant", content: [{ type: "output_text", text }] });
const call = (call_id: string, name: string, args: Item) => ({ type: "function_call", call_id, name, arguments: JSON.stringify(args) });
const event = (type: string, data: Item) => `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
const completed = (output: Item[], usage?: Item) => event("response.completed", { response: { status: "completed", output, ...(usage ? { usage } : {}) } });

async function setup(sources: Array<string | ((res: http.ServerResponse) => void)>) {
  const dir = await mkdtemp(join(tmpdir(), "deepseek-acp-process-"));
  cleanup.push(() => rm(dir, { recursive: true, force: true }));
  await mkdir(join(dir, "workspace"));
  const cwd = await realpath(join(dir, "workspace"));
  const sessions = join(dir, "sessions");
  const requests: Item[] = [], authorizations: Array<string | undefined> = [];
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    requests.push(JSON.parse(Buffer.concat(chunks).toString()));
    authorizations.push(req.headers.authorization);
    const source = sources.shift();
    if (source === undefined) { res.writeHead(500); res.end("Unexpected request"); return; }
    res.writeHead(200, { "content-type": "text/event-stream" });
    if (typeof source === "function") source(res);
    else res.end(source);
  });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  cleanup.push(() => new Promise<void>(resolve => { server.close(() => resolve()); server.closeAllConnections(); }));
  const upstreamUrl = new URL(`http://127.0.0.1:${(server.address() as AddressInfo).port}/responses`);
  const upstream = vi.fn((_url: URL, options: http.RequestOptions) => http.request(upstreamUrl, options));
  const relay: DeepSeekRelayLease = await createDeepSeekRelay(apiKey, upstream);
  cleanup.push(() => relay.release());
  const backends: InstanceType<typeof DeepSeekBackend>[] = [];
  async function open() {
    const backend = new DeepSeekBackend(relay, sessions);
    backends.push(backend);
    const spawn = vi.spyOn(backend, "spawn");
    const log = vi.fn();
    const client = new AcpClient({ cliPath: "unused-no-provider-cli", cwd, backend, log,
      env: { ...process.env, DEEPSEEK_API_KEY: apiKey, deepseek_other_secret: apiKey },
      timeouts: { requestTimeoutMs: 5000 } });
    cleanup.push(() => client.dispose());
    const initialized = vi.fn();
    client.on("initialized", initialized);
    await client.start();
    expect(initialized).toHaveBeenCalledWith(expect.objectContaining({ agentInfo: { name: "deepseek_adapter", version: "1" }, agentCapabilities: expect.objectContaining({ loadSession: true }) }));
    const spec = spawn.mock.results[0].value;
    expect(spec.args).toEqual([join(buildDir, "deepseek-adapter", "main.mjs")]);
    expect(spec.env.DEEPSEEK_API_KEY).toBeUndefined();
    expect(spec.env.deepseek_other_secret).toBeUndefined();
    expect(JSON.stringify(spec)).not.toContain(apiKey);
    expect(spec.env.DEEPSEEK_RELAY_TOKEN).toBe(relay.token);
    return client;
  }
  return { cwd, sessions, requests, authorizations, upstream, open, backends, relay };
}

describe("compiled DeepSeek ACP process", () => {
  it("streams reasoning, reads through ACP, continues tools and cold-loads canonical history and model/effort", async () => {
    const reasoning = { type: "reasoning", id: "reason-1", content: [{ type: "reasoning_text", text: "Inspect the file." }] };
    const tool = call("read-1", "read_file", { path: "input.txt" });
    const s = await setup([
      event("response.reasoning_text.delta", { output_index: 0, content_index: 0, delta: "Inspect the file." }) + completed([reasoning, tool]),
      event("response.output_text.delta", { output_index: 0, content_index: 0, delta: "File checked." }) + completed([message("File checked.")]),
      completed([message("History retained.")]),
    ]);
    await writeFile(join(s.cwd, "input.txt"), "content supplied by ACP");
    const client = await s.open();
    const promptResult = vi.spyOn(s.backends[0], "normalizePromptResult");
    const read = vi.fn((path: string) => readFile(path, "utf8"));
    client.fsRead = read;
    const thoughts: string[] = [], text: string[] = [], tools: Item[] = [];
    client.on("thoughtChunk", chunk => thoughts.push(chunk));
    client.on("messageChunk", chunk => text.push(chunk));
    client.on("toolCall", tool => tools.push(tool));
    const { sessionId } = await client.newSession();
    expect(client.currentModeId).toBe("agent");
    expect(client.currentModelSupportsEffort()).toBe(true);
    expect(client.availableModels[0].reasoningEfforts).toEqual(["none", "low", "high", "max"]);
    await client.setReasoningEffort("max");
    await client.setModel("deepseek-v4-pro");
    await client.prompt("Check input.txt");
    expect(promptResult).toHaveBeenCalledWith(expect.objectContaining({ stopReason: "end_turn" }));
    expect(thoughts.join("")).toBe("Inspect the file.");
    expect(text.join("")).toBe("File checked.");
    expect(tools).toContainEqual(expect.objectContaining({ toolCallId: "read-1" }));
    expect(read).toHaveBeenCalledOnce();
    expect(read).toHaveBeenCalledWith(join(s.cwd, "input.txt"));
    const history = [
      { type: "message", role: "user", content: [{ type: "input_text", text: "Check input.txt" }] },
      reasoning, tool, { type: "function_call_output", call_id: "read-1", output: "content supplied by ACP" },
    ];
    expect(s.requests[1]).toMatchObject({ model: "deepseek-v4-pro", reasoning: { effort: "max" }, input: history });
    const saved = JSON.parse(await readFile(join(s.sessions, `${sessionId}.json`), "utf8"));
    expect(saved).toMatchObject({ model: "deepseek-v4-pro", effort: "max", history: [...history, message("File checked.")] });
    await client.dispose();
    const cold = await s.open();
    const replay: string[] = [], replayThoughts: string[] = [], replayTools: Item[] = [];
    cold.on("messageChunk", text => replay.push(text));
    cold.on("thoughtChunk", text => replayThoughts.push(text));
    cold.on("toolCall", tool => replayTools.push(tool));
    expect((await cold.listSessions()).sessions.map(row => row.sessionId)).toEqual([sessionId]);
    await cold.loadSession(sessionId);
    expect(cold.currentModelId).toBe("deepseek-v4-pro");
    expect(cold.currentReasoningEffort).toBe("max");
    expect(replay.join("")).toBe("File checked.");
    expect(replayThoughts.join("")).toBe("Inspect the file.");
    expect(replayTools).toContainEqual(expect.objectContaining({ toolCallId: "read-1" }));
    expect(s.requests).toHaveLength(2);
    await cold.prompt("Continue");
    expect(s.requests[2]).toMatchObject({ model: "deepseek-v4-pro", reasoning: { effort: "max" } });
    expect(s.requests[2].input.slice(0, -1)).toEqual(saved.history);
    for (const request of s.requests) {
      expect(request).not.toHaveProperty("previous_response_id");
      expect(request).not.toHaveProperty("conversation");
    }
    expect(s.authorizations).toEqual(s.requests.map(() => `Bearer ${apiKey}`));
    expect(s.upstream).toHaveBeenCalledTimes(3);
    expect(s.upstream.mock.calls.every(([url]) => url.href === "https://api.deepseek.com/responses")).toBe(true);
    expect(JSON.stringify(saved)).not.toContain(apiKey);
    expect(JSON.stringify(saved)).not.toContain(s.relay.token);
  });

  it("cancels a live Responses stream through ACP, closes upstream and admits another prompt", async () => {
    let started!: () => void, closed!: () => void;
    const streaming = new Promise<void>(resolve => { started = resolve; });
    const disconnected = new Promise<void>(resolve => { closed = resolve; });
    const s = await setup([res => {
      res.once("close", closed);
      res.write(event("response.output_text.delta", { output_index: 0, content_index: 0, delta: "Partial" }));
      started();
    }, completed([message("Recovered.")])]);
    const client = await s.open();
    const promptResult = vi.spyOn(s.backends[0], "normalizePromptResult");
    await client.newSession();
    const running = client.prompt("Wait for cancellation");
    await streaming;
    expect(await client.cancel("process-test")).toBe(true);
    await running;
    expect(promptResult).toHaveBeenLastCalledWith(expect.objectContaining({ stopReason: "cancelled" }));
    await disconnected;
    await client.prompt("Try again");
    expect(promptResult).toHaveBeenLastCalledWith(expect.objectContaining({ stopReason: "end_turn" }));
    expect(s.requests).toHaveLength(2);
  });

  it("runs an approved command via real ACP terminal handlers and releases it", async () => {
    vi.stubEnv("DEEPSEEK_API_KEY", apiKey);
    vi.stubEnv("DEEPSEEK_RELAY_TOKEN", "ambient-capability-must-not-reach-tools");
    const script = "process.stdout.write(JSON.stringify({args:process.argv.slice(1),secrets:Object.keys(process.env).filter(k=>/^DEEPSEEK_/i.test(k))}))";
    const args = ["-e", script, "two words", "$(not-a-shell)", "x && y"];
    const s = await setup([
      completed([call("command-1", "run_command", { command: process.execPath, args })], { input_tokens: 100, output_tokens: 20, total_tokens: 120,
        input_tokens_details: { cached_tokens: 30 }, output_tokens_details: { reasoning_tokens: 5 } }),
      completed([message("Command checked.")], { input_tokens: 140, output_tokens: 10, total_tokens: 150,
        input_tokens_details: { cached_tokens: 40 }, output_tokens_details: { reasoning_tokens: 3 } }),
    ]);
    const client = await s.open();
    const promptResult = vi.spyOn(s.backends[0], "normalizePromptResult");
    const manager = new TerminalManager();
    cleanup.push(() => manager.disposeAll());
    client.terminal = manager.ownedBy(client);
    const create = vi.spyOn(client.terminal, "create"), release = vi.spyOn(client.terminal, "release");
    const permissions: PermissionRequest[] = [];
    client.on("permissionRequest", (request: PermissionRequest) => {
      permissions.push(request);
      client.respondPermission(request.id, "allow");
    });
    await client.newSession();
    const result = await client.prompt("Run the command");
    expect(result).toMatchObject({ totalTokens: 150, inputTokens: 140, outputTokens: 10, cachedReadTokens: 40, reasoningTokens: 3,
      usage: { inputTokens: 240, outputTokens: 30, totalTokens: 270, cachedReadTokens: 70, reasoningTokens: 8 } });
    expect(promptResult).toHaveBeenCalledWith(expect.objectContaining({ stopReason: "end_turn" }));
    expect(permissions).toHaveLength(1);
    expect(permissions[0].toolCall).toMatchObject({ toolCallId: "command-1", kind: "execute" });
    expect(create).toHaveBeenCalledOnce();
    expect(create.mock.calls[0][0]).toMatchObject({ command: process.execPath, args, cwd: s.cwd, directArgv: true });
    expect(create.mock.calls[0][0]).not.toHaveProperty("env");
    const output = JSON.parse(s.requests[1].input.find((item: Item) => item.type === "function_call_output").output);
    expect(output).toMatchObject({ exitCode: 0, truncated: false });
    expect(JSON.parse(output.output)).toEqual({ args: args.slice(2), secrets: [] });
    expect(release).toHaveBeenCalledOnce();
    const terminalId = release.mock.calls[0][0];
    expect(() => manager.output(terminalId)).toThrow("unknown terminalId");
    expect(manager.anyRunning()).toBe(false);
  });
});
