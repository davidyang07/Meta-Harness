# Historical design records

These documents predate the implementation. They are kept for the
reasoning behind the architecture, not as a description of current
behaviour. Their aspirational wording is not a claim about the shipped
system: for what is implemented and how each capability is validated,
read [`docs/CAPABILITIES.md`](../CAPABILITIES.md) and
[`docs/INTERFACES.md`](../INTERFACES.md).

| File | What it was |
|---|---|
| `ARCHITECTURE_SECTION_1.md` | The original two-tier architecture and its locked decisions |
| `relay_metaharness_v7.md` | The v7 build document (properties, demo arc, scope) |
| `relay_v7_appendix_a_worktrees.md` | Concurrent branches via `asyncio.create_task` |
| `relay_v7_appendix_b_metaharness_internals.md` | Notes on the Stanford reference framework and the SKILL.md skeleton |
| `relay_v7_appendix_c_inner_loop.md` | The five-phase inner agent, its six tools and override points |
| `stitch_mcp.md` | Third-party setup notes for the Google Stitch MCP server, used while designing the dashboard |
| `frontend-design-v1.md`, `frontend-mockup-v1.html` | The hackathon dashboard design and its static mockup, replaced by `frontend/DESIGN.md` |
