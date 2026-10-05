<p align="center">
  <img src="docs/banner.png" alt="Turbohaul-Manager: a GPU rig hauling inference traffic" width="640">
</p>

# Turbohaul-Manager

**One GPU, a whole fleet of AI agents.** Turbohaul-Manager is a self-hosted inference server for local GGUF models on NVIDIA GPUs (Blackwell included). It speaks the Ollama and OpenAI APIs, queues every agent's requests on one card, and keeps each conversation's work warm. A follow-up turn picks up where the last one left off instead of making the model re-read the whole conversation.

**Why it matters:** a large context that takes about 327 s to re-read comes back in about 0.5 s. In one measured restore after a full unload, 154,647 tokens were reused with only 29 new tokens to process.

- 🚦 **Fast Lane:** decide which clients, and which kinds of work, get served first. Off by default.
- ♻️ **Warm conversations:** each conversation's computed context (its KV cache) is saved and restored across pauses, model swaps and idle unloads.
- 🔌 **Drop-in API:** any Ollama or OpenAI client connects with two lines of config.
- 🚛 **Many agents, one card:** requests queue cleanly, the warm model is kept when it can be, and several models can stay loaded at once (`max_parallel_sidecars`).
- 📊 **Web dashboard:** watch what is loaded, running and queued, and manage models, manifests and the blob store from the browser.
- 🛡️ **Safety first:** refuses to load a model when VRAM, RAM, CPU or disk wait would put the host at risk.
- 📦 **Fully vendored:** the engine (a [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) fork of llama.cpp), the Python wheels and the web UI live in this one repository.

![The Turbohaul-Manager dashboard: two models serving live traffic side by side](docs/dashboard-live.png)

## Quick start

Turbohaul is built from source; no prebuilt image is published.

```bash
git clone https://github.com/MrTrenchTrucker/turbohaul-manager.git
cd turbohaul-manager
docker build -f Dockerfile.engine-src -t turbohaul-manager:v0.8.0 .

docker run --gpus all -p 127.0.0.1:11401:11401 \
    -v $(pwd)/state:/var/lib/turbohaul \
    -v $(pwd)/models:/var/lib/turbohaul/import-staging \
    turbohaul-manager:v0.8.0
```

Then open **http://127.0.0.1:11401/ui**.

- ⚠️ **No built-in login.** Turbohaul performs no authentication, so the bind address is the security boundary. Keep `127.0.0.1` unless you have read [Security model & hardening](ARCHITECTURE.md).
- 💾 **Keep the `state` mount.** It holds the database, the manifests and the model blobs. Without it they are lost when the container is removed. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md).
- 🖥️ **Hardware:** three tiers (Minimum, Recommended, High) are in [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md).
- 🧰 **Build options:** `Dockerfile.engine-src` builds fully offline from the vendored sources for CUDA architectures 89 and 120 (add `--build-arg CUDA_ARCH="<list>"` to change them). `Dockerfile.cuda-multi` covers Turing through Blackwell, but installs the Python and frontend dependencies from the network.

## Connect your agent

Point any OpenAI-compatible client (an AI harness, an agent framework, the OpenAI SDK, an Ollama client) at Turbohaul:

```yaml
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy   # no authentication is performed
```

The defaults are tuned for multi-tool-call agent loops: streaming pass-through, tool-call forwarding on both APIs, recovery of tool calls a model writes as plain text, and warm reuse for same-thread follow-ups.

- **Full guide:** [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md)
- **Agent skills** (for an AI agent that sets up or runs Turbohaul for you): [skills/](skills/)

## 🚦 Fast Lane: deciding who goes next

![Fast Lane: the rig running the priority lane past queued traffic](docs/fastlane-banner.png)

When several agents share one GPU, somebody has to go first. By default requests are served in arrival order. Fast Lane lets you say it should be *this* client, and *this* kind of work (main conversation, curator, compression, sub-agent), ahead of the rest.

- **It never interrupts a reply.** It only chooses which waiting request goes next. A lower-priority idle model may be unloaded to make room, but only between turns.
- **Nobody starves.** Unlisted traffic has a wall-clock fairness floor (`fastlane.max_normal_wait_s`).
- **Easy to set up:** turn it on at **Settings → General → Fast Lane**, then pick clients from the **Discovered** list at **Queue → Fast Lane**.

![The Fast Lane screen: priority rules per client and per kind of work, with the Discovered list below](docs/fastlane-rules.png)

Full guide: [docs/FAST_LANE.md](docs/FAST_LANE.md)

## More under the hood

- **Several models, or several copies of one:** `max_parallel_sidecars` sets how many engines run box-wide, and `llama_server_flags.parallel` sets context windows per engine. See [docs/SIDECARS_AND_CONTEXT_WINDOWS.md](docs/SIDECARS_AND_CONTEXT_WINDOWS.md).
- **Multi-GPU placement:** assign models to cards explicitly or automatically. See [docs/MULTI_GPU_PLACEMENT.md](docs/MULTI_GPU_PLACEMENT.md).
- **Hybrid (SSM + attention) models** are sized correctly from their real dimensions. See [docs/HYBRID_KV_RATIO.md](docs/HYBRID_KV_RATIO.md).
- **Vision models and speculative decoding**, configured per model. See [docs/VISION_MODELS.md](docs/VISION_MODELS.md) and [docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md).
- **Tool-call recovery** for models that write their tool calls as text. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **Plugins** *(work in progress):* operator-run HTTP services (a transcriber, a document converter) invoked through the API. See [docs/PLUGINS_SETUP.md](docs/PLUGINS_SETUP.md).

## 📚 Documentation

| I want to... | Read |
|---|---|
| Understand how it all fits together | [ARCHITECTURE.md](ARCHITECTURE.md) |
| See every API endpoint | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |
| Use the web dashboard | [docs/frontend/README.md](docs/frontend/README.md) |
| Connect an AI agent | [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md) · [skills/](skills/) |
| Set up Fast Lane | [docs/FAST_LANE.md](docs/FAST_LANE.md) |
| Add and configure a model (backend and dashboard) | [docs/MODELS_AND_MANIFESTS.md](docs/MODELS_AND_MANIFESTS.md) |
| Look up every manifest field and flag | [docs/MODEL_CONFIG_REFERENCE.md](docs/MODEL_CONFIG_REFERENCE.md) · [docs/TURBOQUANT_FLAGS.md](docs/TURBOQUANT_FLAGS.md) |
| Understand cache reuse | [docs/KV_CACHE_MATCHING.md](docs/KV_CACHE_MATCHING.md) · [docs/REASONING_KV_REUSE.md](docs/REASONING_KV_REUSE.md) · [docs/KV_CACHE_RAM_LIFECYCLE.md](docs/KV_CACHE_RAM_LIFECYCLE.md) |
| Plan GPUs and memory | [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) · [docs/SAFETY_GATE_VRAM_MATH.md](docs/SAFETY_GATE_VRAM_MATH.md) · [docs/MULTI_GPU_PLACEMENT.md](docs/MULTI_GPU_PLACEMENT.md) |
| Share one GPU between many agents | [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md) · [docs/DEPLOYMENT_PATTERNS.md](docs/DEPLOYMENT_PATTERNS.md) |
| Run it in production | [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md) |
| Add plugins *(WIP)* | [docs/PLUGINS_SETUP.md](docs/PLUGINS_SETUP.md) · [docs/PLUGIN_REFERENCE.md](docs/PLUGIN_REFERENCE.md) |
| See what changed | [CHANGELOG.md](CHANGELOG.md) |

## Known issues

- 💾 **Saving the KV cache to disk when a model unloads is a work in progress and does not currently work.** The RAM tier works, so cache reuse across model swaps is unaffected.
- ⬆️ **Upgrading from v0.7.0?** One retired setting stops the manager from starting until you remove it. Read the *Compatibility* notes in [CHANGELOG.md](CHANGELOG.md) before you restart.

## Contributing, credits and license

- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Contributors:** [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **License:** MIT (see [LICENSE](LICENSE)). Third-party dependencies are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), all under MIT or MIT-compatible permissive licenses.
