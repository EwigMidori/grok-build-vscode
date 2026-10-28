# DeepSeek

AFK Pilot includes an ACP agent for DeepSeek's
[Responses API](https://api-docs.deepseek.com/guides/responses_api).
It does not need Codex, DSH, or another external CLI.

## Connect

1. Get an API key from [DeepSeek Platform](https://platform.deepseek.com/).
2. On the computer running the workspace, open **Settings > Providers > DeepSeek**.
3. Paste the key and choose **Use key**.
4. Start a new conversation and choose a DeepSeek model.

The key is stored in the host's secure secret store. A saved key can be reused
after Disconnect; **AFK Pilot: Log Out** deletes it. An invalid key is reported
when a request reaches DeepSeek. Key entry is unavailable from a linked phone,
but an already-connected conversation can be used remotely.

## Models and tools

| Model | Inputs |
|---|---|
| `deepseek-flash` | Text and images |
| `deepseek-v4-pro` | Text |

Both offer `none`, `low`, `high`, and `max` reasoning efforts. A conversation
containing images cannot switch to the text-only model. Replies and reasoning
stream into the existing chat and thinking views.

The bundled agent can read workspace text files, replace workspace files, and
run executables with separate arguments. File tools reject outside paths and
symlinks escaping the workspace. Writes and commands require approval. Commands
are not sandboxed; an approved executable or interpreter has your account's
normal filesystem and network access. Review the full command and arguments.

Agent is the only mode. Plan, Auto accept, steering, MCP, compaction, and history
deletion are not offered. Existing providers keep their own tools and modes.

## History and transport

The endpoint is `POST https://api.deepseek.com/responses`, not Chat Completions.
DeepSeek Responses is stateless: every model call includes the full canonical
history, including raw reasoning items, function calls and their results. The
adapter persists that same history for cold resume, independently of whether
thinking is visible. Interrupted calls receive paired failure results and are
not silently rerun on resume.

Histories live in `deepseek/sessions` under the host's global storage directory.
The standalone adapter uses `~/.afkpilot/deepseek/sessions` unless
`DEEPSEEK_SESSION_DIR` is supplied. Neither history nor the adapter environment
contains the saved API key. A host-owned loopback relay replaces a short-lived
capability with the key only when forwarding to the fixed DeepSeek endpoint.
See [Privacy](privacy.md#deepseek-conversations).

## Verification

The CLI-free tests use simulated Responses streams and loopback transports.
They cover text/reasoning streaming, tool continuation, canonical cold resume,
permissions, workspace containment, cancellation, relay failures, and the real
ACP executable/argv path. A live API request still requires your own key and
may incur DeepSeek usage charges.
