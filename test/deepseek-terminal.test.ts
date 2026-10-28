import * as os from "node:os";
import { realpathSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AcpClient } from "../src/acp";
import type { AcpBackend } from "../src/acp-backend";
import { DeepSeekBackend } from "../src/deepseek-backend";
import { CodexBackend } from "../src/codex-backend";
import { ClaudeBackend } from "../src/claude-backend";
import { MuseBackend } from "../src/muse-backend";
import { AgyBackend } from "../src/agy-backend";
import { TerminalManager } from "../src/terminal-manager";

afterEach(() => vi.unstubAllEnvs());

function clientWithTerminal(backend?: AcpBackend) {
  const client = new AcpClient({ cliPath: "unused", cwd: process.cwd(), backend, log: () => {} });
  const replies: any[] = [];
  (client as any).proc = { stdin: { writable: true, write: (line: string) => { replies.push(JSON.parse(line)); } } };
  const manager = new TerminalManager();
  client.terminal = manager.ownedBy(client);
  let id = 0;
  return { client, manager, async request(method: string, params: unknown) {
    const requestId = ++id;
    await (client as any).handleServerRequest({ id: requestId, method, params });
    const reply = replies.find(message => message.id === requestId);
    expect(reply).toBeDefined();
    return reply;
  } };
}

describe("DeepSeek ACP terminal execution", () => {
  it("passes executable argv literally through the ACP handler to a real process", async () => {
    const h = clientWithTerminal(new DeepSeekBackend());
    const create = vi.spyOn(h.manager, "create");
    const literalArgs = ["two words", "", "double\"quote", "single'quote", "$(node --version)", "`node --version`", "$HOME", "%USERNAME%", "*.txt", "a; b", "x && y", "x | y", "x > y"];
    const script = "process.stdout.write(JSON.stringify({args:process.argv.slice(1),cwd:process.cwd()}))";
    try {
      const created = await h.request("terminal/create", { command: process.execPath, args: ["-e", script, ...literalArgs], cwd: os.tmpdir(), directArgv: false });
      expect(created.error).toBeUndefined();
      expect(create).toHaveBeenCalledWith(expect.objectContaining({ directArgv: true, args: ["-e", script, ...literalArgs] }), h.client);
      const { terminalId } = created.result;
      expect((await h.request("terminal/wait_for_exit", { terminalId })).result).toEqual({ exitCode: 0 });
      const output = (await h.request("terminal/output", { terminalId })).result;
      expect(output.exitStatus).toEqual({ exitCode: 0 });
      expect(JSON.parse(output.output)).toEqual({ args: literalArgs, cwd: realpathSync(os.tmpdir()) });
      expect((await h.request("terminal/release", { terminalId })).error).toBeUndefined();
      expect(() => h.manager.output(terminalId)).toThrow("unknown terminalId");
    } finally { h.manager.disposeAll(); }
  });

  it("filters ambient and request-supplied DeepSeek credentials and capabilities on the host", async () => {
    for (const name of ["DEEPSEEK_API_KEY", "DEEPSEEK_RESPONSES_URL", "DEEPSEEK_RELAY_TOKEN", "DEEPSEEK_SESSION_DIR", "deepseek_other_secret"]) vi.stubEnv(name, "ambient-secret");
    const h = clientWithTerminal(new DeepSeekBackend());
    try {
      const created = await h.request("terminal/create", {
        command: process.execPath,
        args: ["-e", "process.stdout.write(JSON.stringify({names:Object.keys(process.env).filter(n=>/^DEEPSEEK_/i.test(n)),kept:process.env.TERMINAL_TEST_SAFE}))"],
        env: [{ name: "DEEPSEEK_API_KEY", value: "override-secret" }, { name: "DeEpSeEk_ReLaY_ToKeN", value: "forged-capability" }, { name: "TERMINAL_TEST_SAFE", value: "kept" }],
      });
      expect(created.error).toBeUndefined();
      const { terminalId } = created.result;
      expect((await h.request("terminal/wait_for_exit", { terminalId })).result.exitCode).toBe(0);
      expect(JSON.parse((await h.request("terminal/output", { terminalId })).result.output)).toEqual({ names: [], kept: "kept" });
      expect(process.env.DEEPSEEK_API_KEY).toBe("ambient-secret");
    } finally { h.manager.disposeAll(); }
  });

  it("reports direct executable spawn failures instead of interpreting a shell command", async () => {
    const h = clientWithTerminal(new DeepSeekBackend());
    try {
      const created = await h.request("terminal/create", { command: `${process.execPath} --version`, args: [] });
      const { terminalId } = created.result;
      expect((await h.request("terminal/wait_for_exit", { terminalId })).result.exitCode).toBe(-1);
      expect((await h.request("terminal/output", { terminalId })).result.output).toContain("[spawn error]");
    } finally { h.manager.disposeAll(); }
  });

  it.each([{}, [1], "--version"])("refuses malformed argv %j through ACP", async args => {
    const h = clientWithTerminal(new DeepSeekBackend());
    try {
      const reply = await h.request("terminal/create", { command: process.execPath, args });
      expect(reply.error.message).toContain("Terminal args must be an array of strings");
      expect(h.manager.anyRunning()).toBe(false);
    } finally { h.manager.disposeAll(); }
  });

  it.each([undefined, new CodexBackend(), new ClaudeBackend(), new MuseBackend(), new AgyBackend()])("does not let another provider select direct argv (%s)", async backend => {
    const h = clientWithTerminal(backend);
    const create = vi.fn(() => ({ terminalId: "fake" }));
    h.client.terminal!.create = create;
    const params = { command: "existing shell command", args: ["unchanged"], env: [{ name: "DEEPSEEK_API_KEY", value: "unchanged" }] };
    await h.request("terminal/create", { ...params, directArgv: true });
    expect(create).toHaveBeenCalledWith(params);
  });
});
