# ClimateTwin Kiro Steering Setup

Copy the `.kiro/steering/` directory into the root of your ClimateTwin repository.

This package also includes the research prompt at:

`docs/research/ClimateTwin_Research_Implementation_Prompt.md`

so the live file reference inside `product.md` resolves immediately.

Recommended next sequence in Kiro:

1. Open the ClimateTwin repository as the workspace.
2. Confirm the six workspace steering files appear under Agent Steering & Skills.
3. Start a new Kiro session so always-included steering is loaded cleanly.
4. Ask Kiro to read the project steering and research prompt and summarize the non-negotiable scientific constraints before generating a spec.
5. Create one master ClimateTwin feature spec, then add hooks after the spec/task structure exists.

Do not paste API keys into any steering file.
