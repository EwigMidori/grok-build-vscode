import * as path from "node:path";
import { EventEmitter } from "node:events";
import { describe, expect, it, vi } from "vitest";
import { DeepSeekBackend, DEEPSEEK_API_KEY_SECRET, normalizeDeepSeekApiKey } from "../src/deepseek-backend";
import { ACP_PROVIDERS, INTERNAL_PROVIDERS, supportsApprovalModeSwitching, supportsAutoAccept, supportsClientMcpServers, supportsCompaction, supportsModeSwitching } from "../src/acp-backend";
import { parseWebviewMsg } from "../src/desktop/webview-msg-validate";
import { parseRelayFrame } from "../src/remote-frames";
import { GrokSidebar } from "../src/sidebar";
import { sessionModes, usesClientAutoAccept } from "../src/mode-prefs";

describe("DeepSeek host integration", () => {
  it("is bundled, registered internally, and opts out of unsupported actions", () => {
    expect(ACP_PROVIDERS).toEqual(["grok", "codex", "claude"]);
    expect(INTERNAL_PROVIDERS).toContain("deepseek");
    for (const supports of [supportsClientMcpServers, supportsCompaction, supportsModeSwitching, supportsApprovalModeSwitching, supportsAutoAccept]) expect(supports("deepseek")).toBe(false);
    expect(sessionModes("deepseek")).toEqual(["agent"]);
    expect(sessionModes("deepseek", true)).toEqual(["agent"]);
    expect(usesClientAutoAccept("deepseek")).toBe(false);
    const host: any = Object.create(GrokSidebar.prototype);
    expect(host.locateProvider("deepseek")).toBe(process.execPath);
    expect(host.createProviderBackend("deepseek")).toBeInstanceOf(DeepSeekBackend);
  });

  it("passes only the loopback capability and storage path, not an inherited key or upstream", () => {
    const backend = new DeepSeekBackend({ responsesUrl: "http://127.0.0.1:1234/responses", token: "ephemeral" }, "/storage/sessions");
    const spec = backend.spawn({ cliPath: "/unused", cwd: "/repo", env: { PATH: "/bin", DEEPSEEK_API_KEY: "real-key", deepseek_api_key: "other", DEEPSEEK_RESPONSES_URL: "https://evil.example" } });
    expect(spec.command).toBe(process.execPath);
    expect(spec.args).toEqual([path.join(__dirname, "../src/deepseek-adapter/main.mjs")]);
    expect(spec.shell).toBe(false);
    expect(spec.env).toEqual({ PATH: "/bin", ELECTRON_RUN_AS_NODE: "1", DEEPSEEK_RESPONSES_URL: "http://127.0.0.1:1234/responses", DEEPSEEK_RELAY_TOKEN: "ephemeral", DEEPSEEK_SESSION_DIR: "/storage/sessions" });
    expect(JSON.stringify(spec)).not.toContain("real-key");
  });

  it("routes only supported native controls and rejects Plan and old effort levels", () => {
    const backend = new DeepSeekBackend();
    expect(backend.setModel("s", "deepseek-v4-pro")).toEqual({ method: "session/set_model", params: { sessionId: "s", modelId: "deepseek-v4-pro" } });
    for (const value of ["none", "low", "high", "max"]) expect(backend.setReasoningEffort("s", undefined, value).params.value).toBe(value);
    for (const value of ["medium", "xhigh"]) expect(() => backend.setReasoningEffort("s", undefined, value)).toThrow();
    expect(backend.setMode("s", "agent").params.modeId).toBe("agent");
    for (const mode of ["yolo", "plan", "auto"]) expect(() => backend.setMode("s", mode)).toThrow();
    expect(backend.steeringCapabilities().supported).toBe(false);
    expect(backend.interject()).toBeNull();
  });

  it("does not switch or remember unsupported modes through host commands", async () => {
    const host: any = Object.create(GrokSidebar.prototype);
    host.rememberGrokConfig = vi.fn();
    const client = { sessionId: "s", setMode: vi.fn() };
    const session = { provider: "deepseek", client, autoApprove: false };
    for (const mode of ["agent", "yolo", "plan", "auto"]) await host.setMode(mode, session);
    expect(client.setMode).not.toHaveBeenCalled();
    expect(host.rememberGrokConfig).not.toHaveBeenCalled();
    expect(session.autoApprove).toBe(false);
  });

  it("separates summed billing from final-call context and preserves raw DeepSeek metadata", () => {
    const backend = new DeepSeekBackend();
    const deepseek = { id: "response-final", usage: { input_tokens: 140, output_tokens: 10, total_tokens: 150,
      input_tokens_details: { cached_tokens: 40 }, output_tokens_details: { reasoning_tokens: 3 } } };
    const usage = { inputTokens: 240, outputTokens: 30, totalTokens: 270, cachedReadTokens: 70, thoughtTokens: 8 };
    const raw = { stopReason: "end_turn", usage, _meta: { deepseek, modelId: "deepseek-flash" } };
    const normalized = backend.normalizePromptResult(raw);
    expect(normalized._meta).toEqual({ deepseek, modelId: "deepseek-flash", inputTokens: 140, outputTokens: 10,
      totalTokens: 150, cachedReadTokens: 40, reasoningTokens: 3,
      usage: { inputTokens: 240, outputTokens: 30, totalTokens: 270, cachedReadTokens: 70, reasoningTokens: 8 } });
    expect(normalized._meta.deepseek).toBe(deepseek);
    expect(normalized.usage).toBe(usage);
    expect(raw._meta).toEqual({ deepseek, modelId: "deepseek-flash" });
  });

  it("keeps billing without inventing context when cancellation has no final response", () => {
    const backend = new DeepSeekBackend();
    const normalized = backend.normalizePromptResult({ stopReason: "cancelled",
      usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120, cachedReadTokens: 30, thoughtTokens: 5 } });
    expect(normalized._meta.totalTokens).toBeUndefined();
    expect(normalized._meta.usage).toEqual({ inputTokens: 100, outputTokens: 20, totalTokens: 120, cachedReadTokens: 30, reasoningTokens: 5 });
    const noUsage = { stopReason: "cancelled", _meta: { marker: true } };
    expect(backend.normalizePromptResult(noUsage)).toBe(noUsage);
  });

  it("preserves measured zero usage and ignores malformed token counts", () => {
    const backend = new DeepSeekBackend();
    const zero = backend.normalizePromptResult({ usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, cachedReadTokens: 0, thoughtTokens: 0 },
      _meta: { deepseek: { usage: { input_tokens: 0, output_tokens: 0 } } } });
    expect(zero._meta.totalTokens).toBe(0);
    expect(zero._meta.usage).toEqual({ inputTokens: 0, outputTokens: 0, totalTokens: 0, cachedReadTokens: 0, reasoningTokens: 0 });
    const invalid = backend.normalizePromptResult({ usage: { inputTokens: NaN, outputTokens: "10", totalTokens: Infinity, cachedReadTokens: -1, thoughtTokens: null },
      _meta: { deepseek: { usage: { input_tokens: 100, output_tokens: Infinity } } } });
    expect(invalid._meta.totalTokens).toBeUndefined();
    expect(Object.values(invalid._meta.usage).every(value => value === undefined)).toBe(true);
  });

  it("paginates canonical history and refuses repeated cursors", async () => {
    const backend = new DeepSeekBackend();
    const request = vi.fn().mockResolvedValueOnce({ sessions: [{ sessionId: "a", cwd: "/repo", _meta: { turnCount: 2 } }], nextCursor: "next" }).mockResolvedValueOnce({ sessions: [{ sessionId: "b", cwd: "/repo" }] });
    expect((await backend.listSessions(request, "/repo")).sessions).toEqual([{ sessionId: "a", cwd: "/repo", _meta: { turnCount: 2 }, turnCount: 2 }, { sessionId: "b", cwd: "/repo" }]);
    expect(request).toHaveBeenLastCalledWith("session/list", { cwd: "/repo", cursor: "next" });
    await expect(backend.listSessions(vi.fn().mockResolvedValue({ sessions: [], nextCursor: "same" }), "/repo")).rejects.toThrow("repeated");
  });

  it("projects saved effort and mode into the host model contract on cold resume", () => {
    const backend = new DeepSeekBackend();
    const raw = { modes: { currentModeId: "agent" }, models: { currentModelId: "deepseek-flash", availableModels: [{ modelId: "deepseek-flash" }] }, configOptions: [{ id: "reasoning_effort", currentValue: "max" }] };
    const normalized = backend.normalizeSessionResponse(raw);
    expect(normalized.models.availableModels[0]._meta).toEqual({ supportsReasoningEffort: true, reasoningEffort: "max", reasoningEfforts: ["none", "low", "high", "max"].map(value => ({ value })) });
    expect(backend.configState(raw, {})).toEqual({ modelId: "deepseek-flash", modeId: "agent", reasoningEffort: "max" });
    expect(raw.models.availableModels[0]).toEqual({ modelId: "deepseek-flash" });
  });

  it("releases a relay even when disposing a client that never spawned", async () => {
    const host: any = Object.create(GrokSidebar.prototype);
    const client: any = new EventEmitter();
    const dispose = vi.fn().mockResolvedValue(undefined);
    client.dispose = dispose;
    const release = vi.fn();
    host.bindDeepseekRelay(client, { release });
    await client.dispose(15);
    expect(release).toHaveBeenCalledOnce();
    expect(dispose).toHaveBeenCalledWith(15);
  });

  it("accepts key entry locally but rejects it on the remote wire", () => {
    for (const key of ["", "a b", "a\nb", "x".repeat(513), 7]) {
      expect(normalizeDeepSeekApiKey(key)).toBeUndefined();
      expect(parseWebviewMsg({ type: "deepseekSignIn", key })).toBeNull();
    }
    expect(normalizeDeepSeekApiKey(" sk-test ")).toBe("sk-test");
    expect(parseWebviewMsg({ type: "deepseekSignIn", key: "sk-test" })).toEqual({ type: "deepseekSignIn", key: "sk-test" });
    expect(parseRelayFrame(JSON.stringify({ t: "msg", clientId: "c", msg: { type: "deepseekSignIn", key: "sk-test" } }))).toBeNull();
  });

  it("stores a locally entered key only in HostSecrets and connects after saving", async () => {
    const host: any = Object.create(GrokSidebar.prototype);
    const secrets = new Map<string, string>();
    host.context = { secrets: { get: async (id: string) => secrets.get(id), store: vi.fn(async (id: string, key: string) => { secrets.set(id, key); }) } };
    host.focused = {}; host.workspaceRoot = () => "/repo";
    host.pool = new Set(); host.remoteClients = { detachedActiveValues: () => [] };
    host.disposeAdapterHistoryClient = vi.fn();
    host.finishProviderLogout = vi.fn(); host.setProviderNeedsLogin = vi.fn(); host.setProviderConnected = vi.fn();
    host.adoptSessionsForConnectedProvider = vi.fn(); host.postLocal = vi.fn();
    await host.onAdmittedMessage({ type: "deepseekSignIn", key: " sk-host-secret " }, "remote");
    expect(secrets.size).toBe(0);
    await host.onAdmittedMessage({ type: "deepseekSignIn", key: " sk-host-secret " }, "local");
    expect(secrets.get(DEEPSEEK_API_KEY_SECRET)).toBe("sk-host-secret");
    expect(host.setProviderConnected).toHaveBeenCalledWith("deepseek", true);
    expect(host.adoptSessionsForConnectedProvider).toHaveBeenCalledWith("deepseek", host.focused);
    expect(host.postLocal).toHaveBeenCalledWith(expect.objectContaining({ state: "provider-connected", provider: "deepseek" }));
    expect(JSON.stringify(host.postLocal.mock.calls)).not.toContain("sk-host-secret");
    host.context.secrets.store.mockRejectedValueOnce(new Error("storage unavailable"));
    host.setProviderConnected.mockClear();
    await host.onAdmittedMessage({ type: "deepseekSignIn", key: "sk-replacement" }, "local");
    expect(host.setProviderConnected).not.toHaveBeenCalled();
    expect(host.postLocal).toHaveBeenLastCalledWith(expect.objectContaining({ reason: "The key could not be saved on this machine." }));
  });

  it("disconnects without deleting the key and explicitly logs out by deleting HostSecrets", async () => {
    const host: any = Object.create(GrokSidebar.prototype);
    host.disconnectProvider = vi.fn(); host.finishProviderLogout = vi.fn();
    host.context = { secrets: { delete: vi.fn() } };
    host.host = { appendLine: vi.fn(), showErrorMessage: vi.fn() };
    const release = vi.fn(); host.deepseekRelays = new Set([{ release }]);
    await host.logout("deepseek");
    expect(host.disconnectProvider).toHaveBeenCalledWith("deepseek", undefined);
    expect(host.context.secrets.delete).not.toHaveBeenCalled();
    await host.logout("deepseek", { signOutCli: true });
    expect(host.context.secrets.delete).toHaveBeenCalledWith(DEEPSEEK_API_KEY_SECRET);
    expect(release).toHaveBeenCalledOnce();
    expect(host.finishProviderLogout).toHaveBeenCalledWith("deepseek", undefined);
  });
});
