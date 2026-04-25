# Agent Workflow IR sideband notes

This fork carries a small amount of agent-workflow-ir-specific observability
code. It is intentionally separate from upstream vLLM scheduling and KV-cache
logic.

## Owned additions

- `vllm/entrypoints/openai/chat_completion/workflow_sideband.py`
  defines the current workflow sideband validation contract.
- `vllm/entrypoints/openai/chat_completion/workflow_test_hook.py`
  records sideband observations for local tests when `WORKFLOW_TEST_HOOK=1`.
- `tests/entrypoints/openai/chat_completion/test_workflow_sideband.py`
  covers the agent-workflow-ir sideband contract and hook safety behavior.

These tests are fork-local guards. They should be treated separately from
upstream vLLM chat-completion behavior tests when reviewing or rebasing the
submodule.

## Upstream behavior guards

The hook and sideband validator must not change vanilla serving behavior:

- with no `vllm_xargs`, validation returns `None`
- with `WORKFLOW_TEST_HOOK` unset, hook recording is a no-op
- debug hook routes are only attached when `WORKFLOW_TEST_HOOK=1`
- invalid workflow sideband is recorded as invalid but does not block requests
- hook file write failures do not propagate into request serving

The sideband is metadata-only. It must not contain runtime-private cache
identity such as stable-prefix handles, prompt fingerprints, backend cache
keys, raw ServePlan atom labels, or DSL/LangGraph provenance.

Current optional metadata includes coarse stable-prefix group hints and cohort
group hints. Cohort hints identify sibling requests that share a declared
workflow field, such as `target_language`, but they do not include the field
value or any backend cache identity.
