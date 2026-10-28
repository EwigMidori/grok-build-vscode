import * as path from "node:path";
import type { AcpBackend, BackendConfigState, BackendSpawnOptions } from "./acp-backend";

export const DEEPSEEK_API_KEY_SECRET = "grok.deepseek.apiKey";
export const MAX_DEEPSEEK_API_KEY_CHARS = 512;

export function normalizeDeepSeekApiKey(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const key = value.trim();
  return key && key.length <= MAX_DEEPSEEK_API_KEY_CHARS && !/[\s\x00-\x1f\x7f]/.test(key) ? key : undefined;
}

export class DeepSeekBackend implements AcpBackend<"deepseek"> {
  readonly provider = "deepseek" as const;
  readonly processName = "DeepSeek ACP adapter";
  readonly usesClientPlanGate = false;

  constructor(private readonly relay?: { responsesUrl: string; token: string }, private readonly sessionDir?: string) {}

  spawn(options: BackendSpawnOptions) {
    const env = { ...options.env };
    // Never inherit an API key or a caller-selected upstream into agent tools.
    for (const key of Object.keys(env)) if (/^DEEPSEEK_/i.test(key)) delete env[key];
    return { command: process.execPath, args: [path.join(__dirname, "deepseek-adapter", "main.mjs")], shell: false,
      env: { ...env, ELECTRON_RUN_AS_NODE: "1", ...(this.relay ? {
        DEEPSEEK_RESPONSES_URL: this.relay.responsesUrl, DEEPSEEK_RELAY_TOKEN: this.relay.token,
      } : {}), ...(this.sessionDir ? { DEEPSEEK_SESSION_DIR: this.sessionDir } : {}) } };
  }
  normalizeSessionResponse(response: any) {
    const effort = response?.configOptions?.find((option: any) => option.id === "reasoning_effort")?.currentValue;
    return { ...response, models: { ...response.models, availableModels: (response.models?.availableModels ?? []).map((model: any) => ({
      ...model, _meta: { ...model._meta, supportsReasoningEffort: true,
        reasoningEffort: effort ?? model._meta?.reasoningEffort,
        reasoningEfforts: ["none", "low", "high", "max"].map(value => ({ value })) },
    })) } };
  }
  normalizePromptResult(result: any) {
    const usage = result?.usage;
    if (!usage || typeof usage !== "object") return result;
    const count = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
    const last = result._meta?.deepseek?.usage;
    const inputTokens = count(last?.input_tokens);
    const outputTokens = count(last?.output_tokens);
    return { ...result, _meta: { ...result._meta,
      // Context is the final call, not the turn's billing sum; cache is included in input.
      totalTokens: inputTokens !== undefined && outputTokens !== undefined ? inputTokens + outputTokens : undefined,
      inputTokens, outputTokens,
      cachedReadTokens: count(last?.input_tokens_details?.cached_tokens),
      reasoningTokens: count(last?.output_tokens_details?.reasoning_tokens),
      usage: {
        inputTokens: count(usage.inputTokens), outputTokens: count(usage.outputTokens),
        totalTokens: count(usage.totalTokens), cachedReadTokens: count(usage.cachedReadTokens),
        reasoningTokens: count(usage.thoughtTokens),
      },
    } };
  }
  normalizeUpdate(update: any, meta: any) { return { update, meta }; }
  normalizePermissionParams(params: any) { return params; }
  setModel(sessionId: string, modelId: string) { return { method: "session/set_model", params: { sessionId, modelId } }; }
  setReasoningEffort(sessionId: string, _modelId: string | undefined, level: string) {
    if (!["none", "low", "high", "max"].includes(level)) throw new Error("Unsupported DeepSeek effort");
    return { method: "session/set_config_option", params: { sessionId, configId: "reasoning_effort", value: level } };
  }
  setMode(sessionId: string, modeId: string) {
    if (modeId !== "agent") throw new Error("DeepSeek offers Agent mode only");
    return { method: "session/set_mode", params: { sessionId, modeId } };
  }
  steeringCapabilities() { return { supported: false, acceptsContent: false }; }
  interject() { return null; }
  steerDelivered() { return false; }
  configState(response: any, fallback: BackendConfigState) {
    return { ...fallback,
      modelId: response?.models?.currentModelId ?? fallback.modelId,
      reasoningEffort: response?.configOptions?.find((option: any) => option.id === "reasoning_effort")?.currentValue ?? fallback.reasoningEffort,
      modeId: response?.modes?.currentModeId ?? fallback.modeId };
  }
  modelSetSucceeded() { return true; }
  async listSessions(request: (method: string, params: any) => Promise<any>, cwd: string) {
    const sessions = [];
    let cursor: string | undefined;
    const seen = new Set<string>();
    do {
      const page = await request("session/list", { cwd, ...(cursor ? { cursor } : {}) });
      if (!Array.isArray(page.sessions)) throw new Error("DeepSeek returned no session catalog");
      sessions.push(...page.sessions.map((entry: any) => ({ ...entry, ...entry._meta })));
      cursor = page.nextCursor || undefined;
      if (cursor && seen.has(cursor)) throw new Error("DeepSeek session cursor repeated");
      if (cursor) seen.add(cursor);
    } while (cursor);
    return { sessions };
  }
  isCredentialError(error: unknown) {
    return /authRequired|authentication required|invalid.api.key|unauthorized|\b401\b/i.test(error instanceof Error ? error.message : String(error));
  }
}
