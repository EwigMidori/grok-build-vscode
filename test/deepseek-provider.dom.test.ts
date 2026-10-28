import { describe, expect, it } from "vitest";
import { bootWebview, click, dispatch } from "./webview-harness";

describe("DeepSeek renderer integration", () => {
  it("requires provider advertisement before drawing an entry or key form", () => {
    const h = bootWebview();
    dispatch(h.window, { type: "providerState", providers: [{ id: "grok", connected: false }] });
    dispatch(h.window, { type: "onboarding", state: "connect-agent" });
    expect(h.doc.querySelector('[data-provider="deepseek"]')).toBeNull();
    dispatch(h.window, { type: "onboarding", state: "deepseek-login", provider: "deepseek" });
    expect(h.doc.querySelector(".onb-deepseek-key")).toBeNull();
  });

  it("offers the bundled provider, submits once, clears the field, and closes on success", () => {
    const h = bootWebview();
    dispatch(h.window, { type: "providerState", providers: [{ id: "deepseek", connected: false }] });
    dispatch(h.window, { type: "onboarding", state: "connect-agent" });
    click(h.window, h.doc.querySelector('[data-provider="deepseek"]'));
    expect(h.posted).toContainEqual({ type: "runGrokLogin", provider: "deepseek" });
    dispatch(h.window, { type: "onboarding", state: "deepseek-login", provider: "deepseek" });
    const wizard = h.doc.querySelector(".connect-wizard-body")!;
    const input = wizard.querySelector(".onb-deepseek-key") as HTMLInputElement;
    expect(input.type).toBe("password");
    input.value = " sk-test ";
    click(h.window, wizard.querySelector('[data-act="deepseekSignInKey"]'));
    expect(h.posted.filter(m => m.type === "deepseekSignIn")).toEqual([{ type: "deepseekSignIn", key: "sk-test" }]);
    expect(input.value).toBe("");
    input.value = "sk-enter";
    input.dispatchEvent(new (h.window as any).KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    expect(h.posted).toContainEqual({ type: "deepseekSignIn", key: "sk-enter" });
    dispatch(h.window, { type: "onboarding", state: "provider-connected", provider: "deepseek" });
    expect(h.doc.querySelector(".connect-wizard-body")).toBeNull();
  });

  it("never offers remote key entry", () => {
    const h = bootWebview({ remote: true });
    dispatch(h.window, { type: "providerState", providers: [{ id: "deepseek", connected: false }] });
    dispatch(h.window, { type: "onboarding", state: "deepseek-login", provider: "deepseek" });
    const panel = h.doc.getElementById("welcome-onboarding")!;
    expect(panel.textContent).toContain("Connect DeepSeek at the computer");
    expect(panel.querySelector("input, .onb-action")).toBeNull();
  });

  it("keeps the advertised model and effort catalog provider-scoped", () => {
    const h = bootWebview();
    dispatch(h.window, { type: "providerState", providers: [{ id: "deepseek", connected: true }] });
    dispatch(h.window, { type: "session", sessionId: "s", provider: "deepseek", currentModelId: "deepseek-flash", models: [
      { modelId: "deepseek-flash", name: "DeepSeek Flash", reasoningEfforts: ["none", "low", "high", "max"] },
      { modelId: "deepseek-v4-pro", name: "DeepSeek V4 Pro", reasoningEfforts: ["none", "low", "high", "max"] },
    ] });
    click(h.window, h.doc.getElementById("gear-btn"));
    expect(h.doc.querySelector(".gear-popover")?.textContent || h.doc.body.textContent).toContain("DeepSeek Flash");
    expect([...h.doc.querySelectorAll(".effort-strip-stop")].map(el => (el as HTMLElement).dataset.effort)).toEqual(["none", "low", "high", "max"]);
  });

  it.each([undefined, ["agent"]])("offers Agent only with mode advertisement %j", modes => {
    const h = bootWebview();
    dispatch(h.window, { type: "providerState", providers: [{ id: "deepseek", connected: true }] });
    dispatch(h.window, { type: "session", sessionId: "s", provider: "deepseek", models: [] });
    if (modes) dispatch(h.window, { type: "modeChanged", modeId: "agent", modes });
    click(h.window, h.doc.getElementById("mode-btn"));
    const rows = [...h.doc.querySelectorAll(".mode-popover-item")];
    expect(rows.map(row => row.querySelector(".mode-item-label")?.textContent)).toEqual(["Agent mode"]);
    click(h.window, rows[0]);
    expect(h.posted.filter(msg => msg.type === "setMode")).toEqual([{ type: "setMode", modeId: "agent" }]);
  });
});
