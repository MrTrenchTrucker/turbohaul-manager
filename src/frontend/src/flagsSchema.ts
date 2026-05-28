// flagsSchema.ts — FE mirror of BE SAFE_LLAMA_FLAGS (manifest.py).
// MUST stay in lockstep with src/turbohaul/manifest.py. Single source of truth.
//
// When BE adds/changes a flag, update here too. Future Wave: codegen this
// from Python via a build step (Approach A — codegen)
// or shared JSON-Schema (Approach B). Today this is hand-mirrored.
//
// Categories drive the UI grouping. Each flag's metadata (type, bounds,
// enum, default, hint) drives the input widget and validation.

export type FlagType =
  | 'int'
  | 'float'
  | 'bool'
  | 'string'
  | 'enum-string'
  | 'int-or-string'      // n_gpu_layers (int OR 'all'/'auto')
  | 'bool-or-enum'       // flash_attn (bool OR 'on'/'off'/'auto')
  | 'chat-template';     // built-in enum OR plain non-Jinja string

export type FlagCategory =
  | 'Common'
  | 'Performance'
  | 'KV Cache'
  | 'Context / RoPE / YaRN'
  | 'MoE / Multi-GPU'
  | 'Sampling'
  | 'Reasoning'
  | 'Speculative / MTP'
  | 'Chat / Template'
  | 'Server'
  | 'Debug';

export interface FlagSpec {
  name: string;
  type: FlagType;
  category: FlagCategory;
  hint: string;
  default?: number | string | boolean;
  bounds?: [number, number];    // (lo, hi) for int/float
  enumValues?: readonly string[]; // for enum-string / bool-or-enum / chat-template
  primary?: boolean;            // featured at top
}

// === Built-in chat_template names (MUST match BE SAFE_CHAT_TEMPLATE_NAMES) ===
export const SAFE_CHAT_TEMPLATE_NAMES: readonly string[] = [
  'chatml', 'llama2', 'llama3', 'llama3.1', 'llama3.2', 'llama3.3',
  'gemma', 'gemma2', 'gemma3', 'gemma4',
  'mistral', 'mistral-v1', 'mistral-v3', 'mistral-v3-tekken', 'mistral-v7',
  'phi3', 'phi4',
  'deepseek', 'deepseek2', 'deepseek-r1',
  'qwen', 'qwen2', 'qwen2.5', 'qwen3', 'qwen3.5', 'qwen3.6',
  'command-r', 'command-r-plus',
  'vicuna', 'alpaca', 'zephyr', 'chatglm3', 'chatglm4',
  'openchat', 'orion', 'yi', 'monarch', 'smollm', 'minicpm',
  'exaone3', 'rwkv-world', 'granite', 'qwen3-thinking', 'qwq',
  'default',
];

// === Full flag schema (mirrors BE SAFE_LLAMA_FLAGS + BOUNDS + STRING_ENUMS) ===
// Order within category drives UI rendering order. "primary" entries appear
// in a featured top section.
export const FLAGS_SCHEMA: readonly FlagSpec[] = [
  // === Common (primary, top of editor) ===
  { name: 'ctx_size', type: 'int', category: 'Common', primary: true, default: 4096, bounds: [1, 2_000_000],
    hint: 'Max context window (tokens). KV cache scales linearly. 4K/8K/32K/64K/128K common. Bigger = more VRAM.' },
  { name: 'n_gpu_layers', type: 'int-or-string', category: 'Common', primary: true, default: 999, bounds: [-1, 999],
    enumValues: ['all', 'auto'],
    hint: '999 = all-on-GPU. 0 = CPU-only. -1 = auto. "all"/"auto" string also accepted.' },
  { name: 'n_predict', type: 'int', category: 'Common', primary: true, default: -1, bounds: [-1, 1_000_000],
    hint: 'Max output tokens per request. -1 = unlimited. Per-request max_tokens from client overrides if passed.' },
  { name: 'reasoning_budget', type: 'int', category: 'Common', primary: true, default: -1, bounds: [-1, 1_000_000],
    hint: 'Thinking-token cap (Hermes/Qwen3 preserved-thinking). -1 = unlimited. 0 = thinking disabled. N = cap at N. Setting in manifest LOCKS per-request override; leave -1 to let clients pass thinking_budget_tokens.' },

  // === Performance ===
  { name: 'threads', type: 'int', category: 'Performance', default: -1, bounds: [-1, 256], hint: 'CPU threads for inference. -1 = auto. Match physical core count.' },
  { name: 'threads_batch', type: 'int', category: 'Performance', default: -1, bounds: [-1, 256], hint: 'CPU threads for batch processing. Defaults to threads if -1.' },
  { name: 'threads_http', type: 'int', category: 'Performance', default: -1, bounds: [-1, 256], hint: 'HTTP server thread pool size.' },
  { name: 'parallel', type: 'int', category: 'Performance', default: -1, bounds: [1, 256], hint: 'Number of parallel slots. Turbohaul uses 1 (serial).' },
  { name: 'batch_size', type: 'int', category: 'Performance', default: 2048, bounds: [1, 65536], hint: 'Logical batch size (tokens).' },
  { name: 'ubatch_size', type: 'int', category: 'Performance', default: 512, bounds: [1, 65536], hint: 'Physical microbatch size. Affects throughput vs. memory.' },
  { name: 'keep', type: 'int', category: 'Performance', default: 0, bounds: [-1, 65536], hint: 'Tokens to retain at start of context (system prompt).' },
  { name: 'flash_attn', type: 'bool-or-enum', category: 'Performance', default: 'auto',
    enumValues: ['on', 'off', 'auto', 'enabled', 'disabled'],
    hint: 'Flash-attention. on/off/auto. bool true/false also accepted (legacy).' },
  { name: 'mlock', type: 'bool', category: 'Performance', default: false, hint: 'Lock model in RAM (mlock). Prevents OS paging.' },
  { name: 'no_mmap', type: 'bool', category: 'Performance', default: false, hint: 'Disable mmap of model file. Forces full RAM load.' },
  { name: 'numa', type: 'enum-string', category: 'Performance', default: 'none',
    enumValues: ['none', 'distribute', 'isolate', 'numactl'], hint: 'NUMA placement strategy.' },
  { name: 'swa_full', type: 'bool', category: 'Performance', default: false, hint: 'Full SWA (sliding window attention) cache.' },
  { name: 'no_perf', type: 'bool', category: 'Performance', default: false, hint: 'Disable perf counters.' },
  { name: 'sleep_idle_seconds', type: 'int', category: 'Performance', default: -1, bounds: [-1, 86400], hint: 'Sleep server when idle for N seconds. -1 = never.' },
  { name: 'cache_reuse', type: 'int', category: 'Performance', default: 0, bounds: [0, 65536], hint: 'KV cache reuse window (tokens).' },
  { name: 'no_context_shift', type: 'bool', category: 'Performance', default: false, hint: 'Disable context shifting when full. Stops on overflow instead.' },
  { name: 'slot_prompt_similarity', type: 'float', category: 'Performance', default: 0.1, bounds: [0.0, 1.0], hint: 'Min similarity for slot prompt re-use (0-1).' },
  { name: 'warmup', type: 'bool', category: 'Performance', default: true, hint: 'Warmup model at startup.' },
  { name: 'check_tensors', type: 'bool', category: 'Performance', default: false, hint: 'Verify tensor checksums at load.' },
  { name: 'repack', type: 'bool', category: 'Performance', default: true, hint: 'Repack tensors for current hardware.' },
  { name: 'op_offload', type: 'bool', category: 'Performance', default: true, hint: 'Offload supported ops to GPU.' },
  { name: 'no_host', type: 'bool', category: 'Performance', default: false, hint: 'Disable host buffer usage.' },
  { name: 'direct_io', type: 'bool', category: 'Performance', default: false, hint: 'Use O_DIRECT for model file reads.' },
  { name: 'cont_batching', type: 'bool', category: 'Performance', default: true, hint: 'Continuous batching across requests.' },

  // === KV Cache ===
  { name: 'cache_type_k', type: 'enum-string', category: 'KV Cache', primary: true, default: 'f16',
    enumValues: ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'],
    hint: 'K-cache quantization. f16 = full quality. q4_0 = quarter size. q8_0 = well-tested half size.' },
  { name: 'cache_type_v', type: 'enum-string', category: 'KV Cache', primary: true, default: 'f16',
    enumValues: ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'],
    hint: 'V-cache quantization. Match cache_type_k for symmetric.' },
  { name: 'kv_offload', type: 'bool', category: 'KV Cache', default: true, hint: 'Offload KV cache to GPU.' },
  { name: 'kv_unified', type: 'bool', category: 'KV Cache', default: true, hint: 'Unified KV cache layout (auto by default).' },
  { name: 'cache_idle_slots', type: 'bool', category: 'KV Cache', default: true, hint: 'Keep idle slot KV in cache.' },
  { name: 'cache_prompt', type: 'bool', category: 'KV Cache', default: true, hint: 'Cache prompt prefix across requests.' },
  { name: 'cache_ram', type: 'int', category: 'KV Cache', default: 8192, bounds: [0, 262144], hint: 'KV cache RAM budget (MiB).' },
  { name: 'ctx_checkpoints', type: 'int', category: 'KV Cache', default: 32, bounds: [0, 1024], hint: 'Number of context checkpoints to retain.' },
  { name: 'checkpoint_every_n_tokens', type: 'int', category: 'KV Cache', default: 8192, bounds: [1, 1_000_000], hint: 'Checkpoint cadence (tokens).' },

  // === Context / RoPE / YaRN (long-context tuning) ===
  { name: 'rope_scaling', type: 'enum-string', category: 'Context / RoPE / YaRN', default: 'linear',
    enumValues: ['none', 'linear', 'yarn'], hint: 'RoPE scaling method.' },
  { name: 'rope_scale', type: 'float', category: 'Context / RoPE / YaRN', default: 1.0, bounds: [0.0, 1000.0], hint: 'RoPE frequency scale factor.' },
  { name: 'rope_freq_base', type: 'float', category: 'Context / RoPE / YaRN', default: 0.0, bounds: [0.0, 10_000_000.0], hint: 'RoPE base frequency. 0 = use model default.' },
  { name: 'rope_freq_scale', type: 'float', category: 'Context / RoPE / YaRN', default: 0.0, bounds: [0.0, 100.0], hint: 'RoPE frequency scale.' },
  { name: 'yarn_orig_ctx', type: 'int', category: 'Context / RoPE / YaRN', default: 0, bounds: [0, 2_000_000], hint: 'YaRN original context size.' },
  { name: 'yarn_ext_factor', type: 'float', category: 'Context / RoPE / YaRN', default: -1.0, bounds: [-1.0, 100.0], hint: 'YaRN extrapolation factor.' },
  { name: 'yarn_attn_factor', type: 'float', category: 'Context / RoPE / YaRN', default: -1.0, bounds: [-1.0, 100.0], hint: 'YaRN attention factor.' },
  { name: 'yarn_beta_slow', type: 'float', category: 'Context / RoPE / YaRN', default: -1.0, bounds: [-1.0, 100.0], hint: 'YaRN slow beta.' },
  { name: 'yarn_beta_fast', type: 'float', category: 'Context / RoPE / YaRN', default: -1.0, bounds: [-1.0, 100.0], hint: 'YaRN fast beta.' },

  // === MoE / Multi-GPU ===
  { name: 'cpu_moe', type: 'bool', category: 'MoE / Multi-GPU', default: false, hint: 'Put ALL MoE experts on CPU. For models too big for GPU. Slower.' },
  { name: 'n_cpu_moe', type: 'int', category: 'MoE / Multi-GPU', default: 0, bounds: [0, 256], hint: 'Number of MoE expert layers on CPU. 0 = none. Use for partial offload to fit big MoE on small GPU.' },
  { name: 'split_mode', type: 'enum-string', category: 'MoE / Multi-GPU', default: 'layer',
    enumValues: ['none', 'layer', 'row', 'tensor'], hint: 'Multi-GPU split strategy.' },
  { name: 'main_gpu', type: 'int', category: 'MoE / Multi-GPU', default: 0, bounds: [0, 16], hint: 'Index of main GPU.' },
  { name: 'fit', type: 'enum-string', category: 'MoE / Multi-GPU', default: 'on',
    enumValues: ['on', 'off'], hint: 'Tom\'s Fork auto-mem-fit toggle.' },
  { name: 'fit_ctx', type: 'int', category: 'MoE / Multi-GPU', default: 4096, bounds: [1, 2_000_000], hint: 'Tom\'s Fork fit-ctx target.' },

  // === Sampling ===
  { name: 'temp', type: 'float', category: 'Sampling', primary: true, default: 0.8, bounds: [0.0, 10.0],
    hint: '0.0 = deterministic. 0.7-1.0 common. >2.0 = chaos.' },
  { name: 'top_k', type: 'int', category: 'Sampling', default: 40, bounds: [0, 10000], hint: 'Top-K sampling. 0 = disabled.' },
  { name: 'top_p', type: 'float', category: 'Sampling', primary: true, default: 0.95, bounds: [0.0, 1.0],
    hint: 'Nucleus sampling. 1.0 = no truncation. 0.9-0.95 common.' },
  { name: 'min_p', type: 'float', category: 'Sampling', default: 0.05, bounds: [0.0, 1.0], hint: 'Minimum probability cutoff.' },
  { name: 'typical_p', type: 'float', category: 'Sampling', default: 1.0, bounds: [0.0, 1.0], hint: 'Typical-P sampling (Ollama parity).' },
  { name: 'top_n_sigma', type: 'float', category: 'Sampling', default: -1.0, bounds: [-1.0, 100.0], hint: 'Top-N-sigma sampling. -1 = disabled.' },
  { name: 'repeat_penalty', type: 'float', category: 'Sampling', default: 1.0, bounds: [0.0, 10.0], hint: 'Repetition penalty. 1.0 = none.' },
  { name: 'repeat_last_n', type: 'int', category: 'Sampling', default: 64, bounds: [-1, 65536], hint: 'Window for repeat penalty (tokens).' },
  { name: 'presence_penalty', type: 'float', category: 'Sampling', default: 0.0, bounds: [-10.0, 10.0], hint: 'Presence penalty (Ollama parity).' },
  { name: 'frequency_penalty', type: 'float', category: 'Sampling', default: 0.0, bounds: [-10.0, 10.0], hint: 'Frequency penalty (Ollama parity).' },
  { name: 'seed', type: 'int', category: 'Sampling', default: -1, bounds: [-1, 9223372036854775807], hint: 'Random seed. -1 = random.' },
  { name: 'mirostat', type: 'int', category: 'Sampling', default: 0, bounds: [0, 2], hint: 'Mirostat mode. 0=off, 1=v1, 2=v2.' },
  { name: 'mirostat_lr', type: 'float', category: 'Sampling', default: 0.1, bounds: [0.0, 1.0], hint: 'Mirostat learning rate (eta).' },
  { name: 'mirostat_ent', type: 'float', category: 'Sampling', default: 5.0, bounds: [0.0, 100.0], hint: 'Mirostat target entropy (tau).' },
  { name: 'xtc_probability', type: 'float', category: 'Sampling', default: 0.0, bounds: [0.0, 1.0], hint: 'XTC sampling probability.' },
  { name: 'xtc_threshold', type: 'float', category: 'Sampling', default: 0.1, bounds: [0.0, 1.0], hint: 'XTC threshold.' },
  { name: 'dynatemp_range', type: 'float', category: 'Sampling', default: 0.0, bounds: [0.0, 10.0], hint: 'Dynamic temperature range.' },
  { name: 'dynatemp_exp', type: 'float', category: 'Sampling', default: 1.0, bounds: [0.0, 10.0], hint: 'Dynamic temperature exponent.' },
  { name: 'dry_multiplier', type: 'float', category: 'Sampling', default: 0.0, bounds: [0.0, 10.0], hint: 'DRY sampler multiplier. 0 = disabled.' },
  { name: 'dry_base', type: 'float', category: 'Sampling', default: 1.75, bounds: [1.0, 10.0], hint: 'DRY base.' },
  { name: 'dry_allowed_length', type: 'int', category: 'Sampling', default: 2, bounds: [0, 65536], hint: 'DRY allowed repeat length.' },
  { name: 'dry_penalty_last_n', type: 'int', category: 'Sampling', default: -1, bounds: [-1, 65536], hint: 'DRY penalty window.' },
  { name: 'adaptive_target', type: 'float', category: 'Sampling', default: -1.0, bounds: [-1.0, 100.0], hint: 'Adaptive sampler target.' },
  { name: 'adaptive_decay', type: 'float', category: 'Sampling', default: 0.9, bounds: [0.0, 1.0], hint: 'Adaptive sampler decay.' },
  { name: 'ignore_eos', type: 'bool', category: 'Sampling', default: false, hint: 'Ignore EOS token (model keeps generating).' },

  // === Reasoning (Hermes preserved-thinking) ===
  { name: 'reasoning_format', type: 'enum-string', category: 'Reasoning', default: 'auto',
    enumValues: ['none', 'deepseek', 'deepseek-legacy', 'auto'], hint: 'Reasoning output format (deepseek-r1 style).' },
  { name: 'reasoning', type: 'enum-string', category: 'Reasoning', default: 'auto',
    enumValues: ['on', 'off', 'auto'], hint: 'Reasoning mode.' },

  // === Speculative / MTP (multi-token-prediction; needs GGUF with nextn head — Qwen3.5/3.6) ===
  { name: 'spec_type', type: 'enum-string', category: 'Speculative / MTP', primary: true, default: 'draft-mtp',
    enumValues: ['draft-mtp'],
    hint: 'Speculative decode type. draft-mtp = model bundled multi-token-prediction head (faster decode on Qwen3.5/3.6 GGUFs that carry the nextn head). Composes with TurboQuant cache types.' },
  { name: 'spec_draft_n_max', type: 'int', category: 'Speculative / MTP', default: 3, bounds: [0, 64], hint: 'Max draft tokens proposed per step. MTP default 3. Higher = more speculative, diminishing returns.' },
  { name: 'spec_draft_n_min', type: 'int', category: 'Speculative / MTP', default: 0, bounds: [0, 64], hint: 'Min draft tokens per step.' },
  { name: 'spec_draft_p_min', type: 'float', category: 'Speculative / MTP', default: 0.0, bounds: [0.0, 1.0], hint: 'Min probability to continue drafting (0 = always draft n_max).' },
  { name: 'spec_draft_p_split', type: 'float', category: 'Speculative / MTP', default: 0.0, bounds: [0.0, 1.0], hint: 'Draft tree split probability threshold.' },
  { name: 'spec_draft_ngl', type: 'int', category: 'Speculative / MTP', default: -1, bounds: [-1, 999], hint: 'Draft GPU layers. Bundled MTP head rides the main model; -1 = same as model.' },
  { name: 'spec_draft_backend_sampling', type: 'bool', category: 'Speculative / MTP', default: false, hint: 'Backend-side sampling for the draft path (+perf on some setups).' },

  // === Chat / Template ===
  { name: 'chat_template', type: 'chat-template', category: 'Chat / Template', default: 'default',
    enumValues: SAFE_CHAT_TEMPLATE_NAMES, hint: 'Built-in template name OR plain string. Jinja constructs ({% / {{) REJECTED (F3 SSTI hardening).' },
  { name: 'jinja', type: 'bool', category: 'Chat / Template', default: true, hint: 'Enable Jinja template processing.' },
  { name: 'skip_chat_parsing', type: 'bool', category: 'Chat / Template', default: false, hint: 'Skip chat-template parsing.' },
  { name: 'special', type: 'bool', category: 'Chat / Template', default: false, hint: 'Output special tokens (BOS/EOS/etc).' },
  { name: 'spm_infill', type: 'bool', category: 'Chat / Template', default: false, hint: 'SentencePiece infill mode.' },

  // === Server ===
  { name: 'metrics', type: 'bool', category: 'Server', default: false, hint: 'Enable Prometheus /metrics endpoint.' },
  { name: 'slots', type: 'bool', category: 'Server', default: true, hint: 'Enable /slots endpoint.' },
  { name: 'props', type: 'bool', category: 'Server', default: false, hint: 'Enable /props endpoint (runtime mutate).' },
  { name: 'embeddings', type: 'bool', category: 'Server', default: false, hint: 'Enable /embeddings endpoint.' },
  { name: 'reranking', type: 'bool', category: 'Server', default: false, hint: 'Enable reranking mode.' },
  { name: 'pooling', type: 'enum-string', category: 'Server', default: 'none',
    enumValues: ['none', 'mean', 'cls', 'last', 'rank'], hint: 'Embedding pooling strategy.' },
  { name: 'offline', type: 'bool', category: 'Server', default: false, hint: 'Offline mode (no network).' },

  // === Debug ===
  { name: 'verbose', type: 'bool', category: 'Debug', default: false, hint: 'Verbose logging.' },
  { name: 'log_disable', type: 'bool', category: 'Debug', default: false, hint: 'Disable all logging.' },
  { name: 'log_colors', type: 'enum-string', category: 'Debug', default: 'auto',
    enumValues: ['on', 'off', 'auto'], hint: 'Log colorization.' },
  { name: 'log_prefix', type: 'bool', category: 'Debug', default: false, hint: 'Prefix log lines with timestamp.' },
  { name: 'log_timestamps', type: 'bool', category: 'Debug', default: false, hint: 'Include timestamps in log lines.' },
  { name: 'log_verbosity', type: 'int', category: 'Debug', default: 3, bounds: [0, 4], hint: 'Log verbosity level (0=silent, 4=trace).' },
];

// Category display order
export const CATEGORY_ORDER: readonly FlagCategory[] = [
  'Common',
  'Performance',
  'KV Cache',
  'Context / RoPE / YaRN',
  'MoE / Multi-GPU',
  'Sampling',
  'Reasoning',
  'Speculative / MTP',
  'Chat / Template',
  'Server',
  'Debug',
];

export function getFlagSpec(name: string): FlagSpec | undefined {
  return FLAGS_SCHEMA.find((f) => f.name === name);
}

export function getFlagsByCategory(cat: FlagCategory): FlagSpec[] {
  return FLAGS_SCHEMA.filter((f) => f.category === cat);
}

// DENIED flag categories (informational — for FE error messages when a flag
// is explicitly rejected by BE)
export const DENIED_FLAGS_INFO: Record<string, string> = {
  model: 'Use model_tag + gguf_blob_sha256 (which gates path).',
  alias: 'Aliases are managed by model_tag.',
  rpc: 'RPC mode is disabled (single-process Turbohaul invariant).',
  host: 'Bind host is set by Turbohaul (127.0.0.1).',
  port: 'Port is allocated by Turbohaul slot manager.',
  model_url: 'Network fetch by URL is RCE-class. Use /api/pull-url with DNS-rebind guard.',
  hf_repo: 'HF fetch is rooted through /api/pull (with auth gate).',
  api_key: 'Sidecar auth is Turbohaul-internal.',
  ssl_key_file: 'TLS is terminated at Turbohaul, not sidecar.',
  path: 'CRITICAL: --path sets static-file serve dir → /etc exfil class.',
  tools: 'CRITICAL: --tools enables shell_exec server API → direct RCE.',
  chat_template_file: 'Use chat_template name (built-in enum). File path is RCE-class.',
  grammar_file: 'Grammar files are RCE-class.',
  lora: 'LoRA path is RCE-class. Bake LoRAs into base GGUF instead.',
};
