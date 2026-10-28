import { Readable, Writable } from "node:stream";
import { agent, ndJsonStream } from "@agentclientprotocol/sdk";
import { DeepSeekSession } from "./session.mjs";
import { ResponsesClient } from "./responses.mjs";

let session: DeepSeekSession;
const app = agent()
  .onConnect(connection => {
    session = new DeepSeekSession(connection.client, new ResponsesClient(process.env.DEEPSEEK_RESPONSES_URL || "", process.env.DEEPSEEK_RELAY_TOKEN || ""));
    // ACP tools execute in the host; nevertheless drop these from our environment.
    delete process.env.DEEPSEEK_RELAY_TOKEN;
    delete process.env.DEEPSEEK_RESPONSES_URL;
  })
  .onRequest("initialize", async ({ params }) => {
    session.initialize(params.clientCapabilities);
    return { protocolVersion: 1, agentInfo: { name: "deepseek_adapter", version: "1" }, _meta: { models: session.models() },
      agentCapabilities: { loadSession: true, sessionCapabilities: { list: {} }, promptCapabilities: { image: true, embeddedContext: false } } };
  })
  .onRequest("session/new", ({ params }) => session.newSession(params.cwd, params.mcpServers))
  .onRequest("session/load", ({ params }) => session.loadSession(params.sessionId, params.cwd, params.mcpServers))
  .onRequest("session/list", ({ params }) => session.listSessions(params.cwd ?? undefined, params.cursor))
  .onRequest("session/set_mode", ({ params }) => session.setMode(params.sessionId, params.modeId))
  .onRequest("session/set_model", (value: unknown) => {
    const p = value as Record<string, unknown>;
    if (!p || typeof p.sessionId !== "string" || typeof p.modelId !== "string") throw new Error("Invalid model selection");
    return { sessionId: p.sessionId, modelId: p.modelId };
  }, ({ params }) => session.setModel(params.sessionId, params.modelId))
  .onRequest("session/set_config_option", (value: unknown) => {
    const p = value as Record<string, unknown>;
    if (!p || typeof p.sessionId !== "string" || p.configId !== "reasoning_effort" || typeof p.value !== "string") throw new Error("Invalid effort selection");
    return { sessionId: p.sessionId, value: p.value };
  }, ({ params }) => session.setReasoningEffort(params.sessionId, params.value))
  .onRequest("session/prompt", ({ params }) => session.prompt(params.sessionId, params.prompt))
  .onNotification("session/cancel", ({ params }) => { session.cancel(params.sessionId); });

const connection = app.connect(ndJsonStream(Writable.toWeb(process.stdout) as WritableStream<Uint8Array>, Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>));
const stop = () => { session?.close(); connection.close(); process.stdin.destroy(); };
process.stdin.on("end", stop);
process.stdin.on("error", stop);
process.stdout.on("error", stop);
process.on("SIGTERM", stop);
process.on("SIGINT", stop);
void connection.closed.then(stop);
