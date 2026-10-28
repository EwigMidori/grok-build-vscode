import { randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, readdir, realpath, rename, stat, writeFile, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { RequestError, type AgentContext, type ContentBlock, type ClientCapabilities, type SessionUpdate, type CreateTerminalResponse, type RequestPermissionResponse, type Usage } from "@agentclientprotocol/sdk";
import { buildRequest, EFFORTS, MODELS, ResponsesClient, TOOLS, type Item } from "./responses.mjs";

const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
type Saved = { version: 1; id: string; cwd: string; model: string; effort: string; mode: string; updatedAt: string; title: string; history: Item[] };
const modes = { availableModes: [{ id: "agent", name: "Agent", description: "Ask before writes and commands" }], currentModeId: "agent" };
const invalid = (message: string) => new RequestError(-32602, message);

async function workspaceFile(cwd: string, input: string, writing: boolean): Promise<string> {
  const within = (root: string, path: string) => {
    const rel = relative(root, path);
    return rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
  };
  const path = resolve(cwd, input);
  if (!within(cwd, path) || (process.platform === "win32" && relative(cwd, path).includes(":"))) throw new Error("File is outside the workspace");
  const root = await realpath(cwd);
  let ancestor = path;
  const missing: string[] = [];
  while (true) {
    try {
      const target = await realpath(ancestor);
      if (!within(root, target)) throw new Error("File target is outside the workspace");
      const info = await stat(target);
      if (missing.length ? !info.isDirectory() : !info.isFile()) throw new Error("Not a regular workspace file");
      return join(target, ...missing);
    } catch (error) {
      if (!writing || (error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      // A dangling symlink is not a missing path we may safely create through.
      try { await lstat(ancestor); throw new Error("Unresolved workspace file target"); }
      catch (entryError) { if ((entryError as NodeJS.ErrnoException).code !== "ENOENT") throw entryError; }
      const parent = dirname(ancestor);
      if (parent === ancestor) throw new Error("Workspace file target does not exist");
      missing.unshift(basename(ancestor));
      ancestor = parent;
    }
  }
}

async function cancellable<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) { void pending.catch(() => {}); signal.throwIfAborted(); }
  let abort!: () => void;
  const cancelled = new Promise<never>((_, reject) => {
    abort = () => reject(new Error("Tool cancelled"));
    signal.addEventListener("abort", abort, { once: true });
  });
  try { return await Promise.race([pending, cancelled]); }
  finally { signal.removeEventListener("abort", abort); }
}

function pairInterrupted(history: Item[]) {
  const outputs = new Set(history.filter(i => i.type === "function_call_output").map(i => i.call_id));
  for (const item of [...history]) if (item.type === "function_call" && !outputs.has(item.call_id)) {
    history.push({ type: "function_call_output", call_id: item.call_id, output: "Tool failed or was cancelled before a result was recorded." });
    outputs.add(item.call_id);
  }
}

function validSaved(s: any, id: string): s is Saved {
  if (!s || s.version !== 1 || s.id !== id || !ID.test(id) || typeof s.cwd !== "string" || !isAbsolute(s.cwd)
    || !MODELS.includes(s.model) || !EFFORTS.includes(s.effort) || s.mode !== "agent" || typeof s.title !== "string"
    || typeof s.updatedAt !== "string" || !Number.isFinite(Date.parse(s.updatedAt)) || !Array.isArray(s.history)) return false;
  const calls = new Set<string>(), results = new Set<string>();
  for (const i of s.history) {
    if (!i || typeof i !== "object") return false;
    if (i.type === "function_call") {
      if (typeof i.call_id !== "string" || !i.call_id || calls.has(i.call_id) || typeof i.name !== "string" || typeof i.arguments !== "string") return false;
      calls.add(i.call_id);
    } else if (i.type === "function_call_output") {
      if (!calls.has(i.call_id) || results.has(i.call_id) || typeof i.output !== "string") return false;
      results.add(i.call_id);
    } else if (i.type === "message" || i.type === "reasoning") {
      if (i.type === "message" && !["user", "assistant"].includes(i.role)) return false;
      if (!Array.isArray(i.content) || i.content.some((p: any) => !p || !(["input_text", "output_text", "reasoning_text"].includes(p.type) ? typeof p.text === "string" : p.type === "input_image" && typeof p.image_url === "string"))) return false;
    } else return false;
  }
  return true;
}

/** Session state never contains relay credentials. Tools run exclusively in the ACP host. */
export class DeepSeekSession {
  private sessions = new Map<string, Saved>();
  private active = new Map<string, AbortController>();
  private capabilities: ClientCapabilities = {};
  constructor(private readonly client: Pick<AgentContext, "request" | "notify">,
    private readonly responses: ResponsesClient,
    private readonly directory = process.env.DEEPSEEK_SESSION_DIR || join(homedir(), ".afkpilot", "deepseek", "sessions")) {}

  initialize(capabilities: ClientCapabilities = {}) { this.capabilities = capabilities; }
  models() { return MODELS.map(id => ({ modelId: id, name: id, description: id === "deepseek-flash" ? "Text and images" : "Text only" })); }
  private state(id: string) { const s = this.sessions.get(id); if (!s) throw invalid("Unknown DeepSeek session"); return s; }
  private idle(id: string) { if (this.active.has(id)) throw invalid("DeepSeek session is busy"); return this.state(id); }
  private info(s: Saved) { return { modes, models: { availableModels: this.models(), currentModelId: s.model }, configOptions: this.config(s) }; }
  private config(s: Saved) { return [{ id: "reasoning_effort", name: "Reasoning effort", category: "thought_level" as const, type: "select" as const, currentValue: s.effort, options: EFFORTS.map(value => ({ value, name: value })) }]; }
  private async update(id: string, update: SessionUpdate) { await this.client.notify("session/update", { sessionId: id, update }); }
  private async save(s: Saved) {
    s.updatedAt = new Date().toISOString();
    const temp = join(this.directory, `${s.id}.${randomUUID()}.tmp`);
    try {
      await mkdir(this.directory, { recursive: true, mode: 0o700 });
      await writeFile(temp, JSON.stringify(s), { mode: 0o600, flag: "wx" });
      await rename(temp, join(this.directory, `${s.id}.json`));
    } catch { await unlink(temp).catch(() => {}); throw new RequestError(-32603, "Could not persist DeepSeek session"); }
  }
  async newSession(cwd: string, mcpServers: unknown[] = []) {
    if (!isAbsolute(cwd) || mcpServers.length) throw invalid("DeepSeek needs an absolute workspace and does not support MCP");
    const s: Saved = { version: 1, id: randomUUID(), cwd: resolve(cwd), model: MODELS[0], effort: "high", mode: "agent", updatedAt: new Date().toISOString(), title: "DeepSeek session", history: [] };
    await this.save(s); this.sessions.set(s.id, s);
    return { sessionId: s.id, ...this.info(s) };
  }
  private async read(id: string) {
    if (!ID.test(id)) throw invalid("Invalid DeepSeek session ID");
    let s: unknown;
    try { s = JSON.parse(await readFile(join(this.directory, `${id}.json`), "utf8")); } catch { throw invalid("DeepSeek session could not be loaded"); }
    if (!validSaved(s, id)) throw invalid("Invalid DeepSeek session history");
    pairInterrupted(s.history); return s;
  }
  async listSessions(cwd?: string, cursor?: string | null) {
    if (cursor) throw invalid("DeepSeek session list does not use cursors");
    let files: string[];
    try { files = await readdir(this.directory); } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return { sessions: [] }; throw new RequestError(-32603, "Could not list DeepSeek sessions"); }
    const sessions = [];
    for (const file of files) {
      if (!file.endsWith(".json") || !ID.test(file.slice(0, -5))) continue;
      try { const s = await this.read(file.slice(0, -5)); if (!cwd || resolve(cwd) === s.cwd) sessions.push({ sessionId: s.id, cwd: s.cwd, title: s.title, updatedAt: s.updatedAt }); } catch { /* Corrupt files are not catalog entries. */ }
    }
    sessions.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return { sessions };
  }
  async loadSession(id: string, cwd: string, mcpServers: unknown[] = []) {
    if (this.active.has(id)) throw invalid("DeepSeek session is busy");
    const s = await this.read(id);
    if (!isAbsolute(cwd) || resolve(cwd) !== s.cwd || mcpServers.length) throw invalid("DeepSeek session workspace mismatch or unsupported MCP");
    await this.save(s); this.sessions.set(id, s);
    for (const item of s.history) {
      if (item.type === "message" || item.type === "reasoning") for (const p of item.content) {
        if (p.type === "input_image") {
          const image = /^data:(image\/(?:png|jpeg|gif|webp));base64,(.*)$/.exec(p.image_url);
          if (image && item.role === "user") await this.update(id, { sessionUpdate: "user_message_chunk", content: { type: "image", mimeType: image[1], data: image[2] } });
          continue;
        }
        await this.update(id, { sessionUpdate: item.role === "user" ? "user_message_chunk" : item.type === "reasoning" ? "agent_thought_chunk" : "agent_message_chunk", content: { type: "text", text: p.text } });
      }
      if (item.type === "function_call") await this.update(id, { sessionUpdate: "tool_call", toolCallId: item.call_id, title: item.name, status: "completed", rawInput: this.safeArgs(item.arguments) });
      if (item.type === "function_call_output") await this.update(id, { sessionUpdate: "tool_call_update", toolCallId: item.call_id, status: "completed", content: [{ type: "content", content: { type: "text", text: item.output } }] });
    }
    return this.info(s);
  }
  async setMode(id: string, mode: string) { const s = this.idle(id); if (mode !== "agent") throw invalid("DeepSeek supports Agent approval mode only"); await this.save(s); return {}; }
  async setModel(id: string, model: string) {
    const s = this.idle(id);
    if (!MODELS.includes(model as typeof MODELS[number])) throw invalid("Unknown DeepSeek model");
    if (model === "deepseek-v4-pro" && s.history.some(i => i.content?.some((p: Item) => p.type === "input_image"))) {
      throw invalid("This session contains images; use deepseek-flash");
    }
    const previous = s.model;
    s.model = model;
    try { await this.save(s); } catch (error) { s.model = previous; throw error; }
    return {};
  }
  async setReasoningEffort(id: string, effort: string) {
    const s = this.idle(id);
    if (!EFFORTS.includes(effort as typeof EFFORTS[number])) throw invalid("Unknown DeepSeek effort");
    const previous = s.effort;
    s.effort = effort;
    try { await this.save(s); } catch (error) { s.effort = previous; throw error; }
    return { configOptions: this.config(s) };
  }
  cancel(id: string) { this.active.get(id)?.abort(); }
  close() { for (const controller of this.active.values()) controller.abort(); }
  private safeArgs(value: string): Item { try { const p = JSON.parse(value); return p && typeof p === "object" && !Array.isArray(p) ? p : {}; } catch { return {}; } }

  private async tool(s: Saved, item: Item, signal: AbortSignal): Promise<string> {
    const p = this.safeArgs(item.arguments);
    const request = async (method: any, params: any): Promise<any> => {
      signal.throwIfAborted();
      return cancellable(this.client.request(method, { sessionId: s.id, ...params }, { cancellationSignal: signal }), signal);
    };
    if (item.name === "read_file" || item.name === "write_file") {
      if (typeof p.path !== "string" || !p.path || p.path.includes("\0")) throw new Error("Invalid file path");
      const path = await workspaceFile(s.cwd, p.path, item.name === "write_file");
      if (item.name === "read_file") {
        if (!this.capabilities.fs?.readTextFile) throw new Error("Host file reading unavailable");
        const content = (await request("fs/read_text_file", { path })).content;
        if (typeof content !== "string") throw new Error("Invalid host file response");
        return content;
      }
      if (typeof p.content !== "string" || !this.capabilities.fs?.writeTextFile) throw new Error("Invalid write or host writing unavailable");
      await this.permission(s, item, { path, content: p.content }, signal);
      if (await workspaceFile(s.cwd, p.path, true) !== path) throw new Error("Workspace file target changed during approval");
      await request("fs/write_text_file", { path, content: p.content });
      return "File written.";
    }
    if (item.name !== "run_command" || !this.capabilities.terminal || typeof p.command !== "string" || !p.command.trim()
      || !Array.isArray(p.args) || p.args.some((a: unknown) => typeof a !== "string")) throw new Error("Invalid or unavailable tool");
    await this.permission(s, item, { command: p.command, args: p.args, cwd: s.cwd }, signal);
    const release = async (terminalId: string) => {
      const timeout = AbortSignal.timeout(2000);
      await cancellable(this.client.request("terminal/release", { sessionId: s.id, terminalId }, { cancellationSignal: timeout }), timeout).catch(() => {});
    };
    // No env field: never forward the adapter environment or its relay capability.
    signal.throwIfAborted();
    const creating = this.client.request<CreateTerminalResponse>("terminal/create", { sessionId: s.id, command: p.command, args: p.args, cwd: s.cwd, outputByteLimit: 100000 }, { cancellationSignal: signal });
    // Creation can finish after Stop. Reclaim that late terminal as well.
    const acquired = creating.then(async terminal => {
      if (signal.aborted) {
        await release(terminal.terminalId);
        signal.throwIfAborted();
      }
      return terminal;
    });
    const terminal = await cancellable(acquired, signal);
    try {
      const exit = await request("terminal/wait_for_exit", { terminalId: terminal.terminalId });
      const output = await request("terminal/output", { terminalId: terminal.terminalId });
      return JSON.stringify({ output: output.output, truncated: output.truncated, exitCode: exit.exitCode, signal: exit.signal });
    } finally {
      // Release also kills a still-running command; cleanup must survive cancellation.
      await release(terminal.terminalId);
    }
  }
  private async permission(s: Saved, item: Item, rawInput: Item, signal: AbortSignal) {
    signal.throwIfAborted();
    const response = await cancellable(this.client.request<RequestPermissionResponse>("session/request_permission", { sessionId: s.id,
      toolCall: { toolCallId: item.call_id, title: item.name, kind: item.name === "write_file" ? "edit" : "execute", status: "pending", rawInput },
      options: [{ optionId: "allow", name: "Allow once", kind: "allow_once" }, { optionId: "reject", name: "Reject", kind: "reject_once" }] }, { cancellationSignal: signal }), signal);
    signal.throwIfAborted();
    if (response.outcome.outcome !== "selected" || response.outcome.optionId !== "allow") throw new Error("Tool permission denied");
  }
  async prompt(id: string, prompt: ContentBlock[]) {
    const s = this.idle(id);
    if (!prompt.length) throw invalid("Empty DeepSeek prompt");
    const content = prompt.map(p => {
      if (p.type === "text") return { type: "input_text", text: p.text };
      if (p.type === "image" && s.model === "deepseek-flash" && ["image/png", "image/jpeg", "image/gif", "image/webp"].includes(p.mimeType)) return { type: "input_image", image_url: `data:${p.mimeType};base64,${p.data}` };
      throw invalid("DeepSeek accepts text, and images with deepseek-flash only");
    });
    const controller = new AbortController(), signal = controller.signal;
    const usage: Usage = { totalTokens: 0, inputTokens: 0, outputTokens: 0, thoughtTokens: 0, cachedReadTokens: 0 };
    this.active.set(id, controller);
    s.history.push({ type: "message", role: "user", content });
    if (s.title === "DeepSeek session") s.title = prompt.filter(p => p.type === "text").map(p => p.text).join(" ").slice(0, 100) || s.title;
    try {
      await this.save(s);
      for (let step = 0; step < 64; step++) {
        signal.throwIfAborted();
        const output: Item[] = [];
        let result: Item;
        try {
          const tools = TOOLS.filter(t => t.name === "read_file" ? this.capabilities.fs?.readTextFile : t.name === "write_file" ? this.capabilities.fs?.writeTextFile : this.capabilities.terminal);
          result = await this.responses.run(buildRequest(s.model, s.effort, s.history, tools), output, signal,
            (text, reasoning) => this.update(id, { sessionUpdate: reasoning ? "agent_thought_chunk" : "agent_message_chunk", content: { type: "text", text } }));
        } finally {
          const canonical = output.filter(Boolean);
          for (const item of canonical) {
            if (item.type === "message" && !item.role) item.role = "assistant";
            if (["message", "reasoning"].includes(item.type)) item.content = (item.content ?? []).filter(Boolean);
          }
          // Reject malformed/duplicate calls before they can execute or poison disk history.
          if (!validSaved({ ...s, history: [...s.history, ...canonical] }, id)) throw new RequestError(-32603, "DeepSeek returned invalid response items");
          s.history.push(...canonical);
          pairInterrupted(s.history);
          await this.save(s);
        }
        if (result.usage) {
          const count = (n: unknown) => typeof n === "number" && Number.isSafeInteger(n) && n >= 0 ? n : 0;
          usage.inputTokens += count(result.usage.input_tokens);
          usage.outputTokens += count(result.usage.output_tokens);
          usage.totalTokens += count(result.usage.total_tokens);
          usage.thoughtTokens! += count(result.usage.output_tokens_details?.reasoning_tokens);
          usage.cachedReadTokens! += count(result.usage.input_tokens_details?.cached_tokens);
          await this.update(id, { sessionUpdate: "session_info_update", title: s.title, updatedAt: s.updatedAt, _meta: { deepseek: { usage: result.usage } } });
        }
        const calls = output.filter(i => i?.type === "function_call");
        if (result.status !== "completed") return { stopReason: result.incomplete_details?.reason === "max_output_tokens" ? "max_tokens" as const : "refusal" as const, usage, _meta: { deepseek: result } };
        for (const call of calls) {
          signal.throwIfAborted();
          await this.update(id, { sessionUpdate: "tool_call", toolCallId: call.call_id, title: call.name, status: "in_progress", rawInput: this.safeArgs(call.arguments) });
          let value: string, failed = false;
          try { value = await this.tool(s, call, signal); }
          catch { value = signal.aborted ? "Tool cancelled." : "Tool failed or permission denied."; failed = true; }
          const placeholder = s.history.find(i => i.type === "function_call_output" && i.call_id === call.call_id)!;
          placeholder.output = value;
          await this.save(s);
          await this.update(id, { sessionUpdate: "tool_call_update", toolCallId: call.call_id, status: failed ? "failed" : "completed", content: [{ type: "content", content: { type: "text", text: value } }] });
        }
        if (!calls.length) return { stopReason: "end_turn" as const, usage, _meta: { deepseek: result } };
      }
      return { stopReason: "max_turn_requests" as const, usage };
    } catch (error) {
      if (signal.aborted) return { stopReason: "cancelled" as const, usage };
      if (error instanceof RequestError) throw error;
      throw new RequestError(-32603, error instanceof Error && error.message.startsWith("DeepSeek ") ? error.message : "DeepSeek turn failed");
    } finally { this.active.delete(id); }
  }
}
