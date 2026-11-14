# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/newmrd0003-create/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/newmrd0003-create/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ordered free model aggregator: one zero-config Node 22 server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier providers, in JSON and in server-sent events. It ranks free models by quality rating, fails over on 402/429/5xx, spreads calls across providers under their per-trial budgets, and prints its catalog with `--list`. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts --list
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=coral-otter-98&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
