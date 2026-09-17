/**
 * WebWorker executing genuine in-browser Verdict-ModernBERT decision inference.
 * Powered by ONNX Runtime Web (WebGPU / WASM) and @huggingface/transformers.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.bundle.min.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js";

env.allowRemoteModels = true;

const CACHE_NAME = "verdict-model-cache-v2";

let session = null;
let tokenizer = null;
let promptContract = null;
let activeBackend = "WASM";
let calibratorTemp = null;
let modelSizeBytes = 0;
let initPromise = null;
let inferenceQueue = Promise.resolve();

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

async function fetchWithCacheAndProgress(url, onProgress) {
  let cache = null;
  if (typeof caches !== "undefined") {
    try {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        const buf = await cached.arrayBuffer();
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return buf;
      }
    } catch (e) {
      console.warn("Cache API lookup failed, fetching from network:", e);
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  }

  const contentLengthHeader = res.headers.get("Content-Length");
  const totalBytes = contentLengthHeader ? parseInt(contentLengthHeader, 10) : 0;
  let receivedBytes = 0;

  const reader = res.body.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;
    if (onProgress) onProgress(receivedBytes, totalBytes);
  }

  const combined = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  if (cache) {
    try {
      await cache.put(url, new Response(combined.buffer, {
        headers: { "Content-Type": "application/octet-stream" }
      }));
    } catch (e) {
      console.warn("Failed to store model in Cache API:", e);
    }
  }

  return combined.buffer;
}

async function initEngine(basePath = "artifacts_v2") {
  try {
    // 1. Session leak fix: Release previous session handle if present
    if (session) {
      try {
        await session.release();
      } catch (relErr) {
        console.warn("Error releasing previous session:", relErr);
      }
      session = null;
    }

    // 2. Load prompt contract
    self.postMessage({ type: "PROGRESS", stage: "loading_contract", message: "Loading prompt contract..." });
    const contractRes = await fetch("prompt_contract.json");
    if (!contractRes.ok) {
      throw new Error(`Failed to fetch prompt_contract.json: HTTP ${contractRes.status}`);
    }
    promptContract = await contractRes.json();

    // 3. Load calibrator (No stale fallback: post ERROR on failure)
    self.postMessage({ type: "PROGRESS", stage: "loading_calibrator", message: "Loading model calibrator..." });
    const calRes = await fetch(`${basePath}/calibrator_modernbert.json`);
    if (!calRes.ok) {
      throw new Error(`Calibrator not found at ${basePath}/calibrator_modernbert.json (HTTP ${calRes.status})`);
    }
    const calData = await calRes.json();
    if (!calData.temperature) {
      throw new Error("Invalid calibrator file: missing 'temperature' parameter");
    }
    calibratorTemp = parseFloat(calData.temperature);

    // 4. Load tokenizer
    self.postMessage({ type: "PROGRESS", stage: "loading_tokenizer", message: "Loading ModernBERT BPE tokenizer..." });
    try {
      tokenizer = await AutoTokenizer.from_pretrained(`${basePath}`, { local_files_only: true });
    } catch (e) {
      tokenizer = await AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-base-v2.0");
    }

    // 5. Download model with streaming progress & Cache API
    self.postMessage({ type: "PROGRESS", stage: "loading_model", message: "Downloading ONNX model weights..." });
    let modelUrl = `${basePath}/model_fp16.onnx`;
    let modelBuffer;
    try {
      modelBuffer = await fetchWithCacheAndProgress(modelUrl, (received, total) => {
        self.postMessage({
          type: "PROGRESS",
          stage: "downloading_weights",
          received,
          total,
          percent: total > 0 ? ((received / total) * 100).toFixed(1) : null,
          message: `Downloading model weights: ${(received / (1024 * 1024)).toFixed(1)} MB` + (total > 0 ? ` / ${(total / (1024 * 1024)).toFixed(1)} MB` : "")
        });
      });
    } catch (fp16Err) {
      console.warn("FP16 model not found, trying model.onnx:", fp16Err);
      modelUrl = `${basePath}/model.onnx`;
      modelBuffer = await fetchWithCacheAndProgress(modelUrl, (received, total) => {
        self.postMessage({
          type: "PROGRESS",
          stage: "downloading_weights",
          received,
          total,
          percent: total > 0 ? ((received / total) * 100).toFixed(1) : null,
          message: `Downloading model weights: ${(received / (1024 * 1024)).toFixed(1)} MB` + (total > 0 ? ` / ${(total / (1024 * 1024)).toFixed(1)} MB` : "")
        });
      });
    }

    modelSizeBytes = modelBuffer.byteLength;
    const modelSizeMb = (modelSizeBytes / (1024 * 1024)).toFixed(1);

    // 6. False backend fix: Attempt WebGPU alone first; catch and attempt WASM alone
    let usedProvider = "WASM";
    if (navigator.gpu) {
      try {
        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all"
        });
        usedProvider = "WebGPU";
      } catch (gpuErr) {
        console.warn("WebGPU initialization failed, falling back to WASM:", gpuErr);
      }
    }

    if (!session) {
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
      usedProvider = "WASM";
    }

    activeBackend = usedProvider;

    // 7. Warm-up inference
    const warmupPrompt = `${promptContract.label_marker}sample${promptContract.label_marker}${promptContract.abstention_description}${promptContract.sep_marker}warmup`;
    const tokens = await tokenizer(warmupPrompt, { truncation: true, max_length: 1024 });
    await session.run({
      input_ids: tokens.input_ids,
      attention_mask: tokens.attention_mask
    });

    self.postMessage({
      type: "READY",
      backend: activeBackend,
      temperature: calibratorTemp,
      modelSizeMb: Number(modelSizeMb)
    });
  } catch (err) {
    self.postMessage({
      type: "ERROR",
      message: `Initialization failed: ${err.message || err}`
    });
  }
}

async function runInference(payload) {
  const { id, text, question, candidates, threshold } = payload;
  const startTime = performance.now();

  if (!session || !tokenizer || !promptContract) {
    throw new Error("Engine not initialized");
  }

  if (candidates.length > promptContract.max_candidates) {
    throw new Error(`Candidate count ${candidates.length} exceeds maximum model capacity of ${promptContract.max_candidates}`);
  }

  // 1. Format prompt strictly from prompt_contract.json
  const t0 = performance.now();
  const resolvedQuestion = question || promptContract.default_question;
  const formattedText = promptContract.input_template
    .replace("{question}", resolvedQuestion)
    .replace("{context}", text);

  const labelPrefix = candidates.map(c => `${promptContract.label_marker}${c.desc}`).join("");
  const prompt = `${labelPrefix}${promptContract.sep_marker}${formattedText}`;

  // 2. Tokenize with explicit bounds
  const tokenized = await tokenizer(prompt, { truncation: true, max_length: 1024 });
  const tokenizationMs = performance.now() - t0;
  const tokenCount = tokenized.input_ids.dims[1];
  const isTruncated = tokenCount >= 1024;

  // 3. Run ONNX model
  const t1 = performance.now();
  const feeds = {
    input_ids: tokenized.input_ids,
    attention_mask: tokenized.attention_mask
  };

  const results = await session.run(feeds);
  const inferenceMs = performance.now() - t1;

  // 4. Readback and slice candidate logits
  const t2 = performance.now();
  const rawOutput = results.logits.data; // Float32Array
  const numCandidates = candidates.length;
  const candidateLogits = Array.from(rawOutput.slice(0, numCandidates));

  // 5. Calibrate logits with temperature T
  const scaledLogits = candidateLogits.map(l => l / calibratorTemp);

  // 6. Stable Softmax
  const maxLogit = Math.max(...scaledLogits);
  const exps = scaledLogits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);
  const postprocessingMs = performance.now() - t2;
  const totalMs = performance.now() - startTime;

  // 7. Find winner & policy action
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const selectedCandidate = candidates[maxIdx];
  const selectedProb = probs[maxIdx];
  const isAbstention = selectedCandidate.id === promptContract.abstention_id;
  const isAutonomous = !isAbstention && selectedProb >= threshold;

  let policyAction = "";
  if (isAbstention) {
    policyAction = "route_tier1_support";
  } else if (isAutonomous) {
    policyAction = `execute_${selectedCandidate.id}`;
  } else {
    policyAction = "escalate_supervisor_review";
  }

  const isCalibratedScope = (numCandidates === 5);
  const probField = isCalibratedScope ? "calibrated_probabilities" : "probabilities";

  // Build verifiable decision receipt
  const receipt = {
    receipt_version: "2.0.0",
    model_id: "Verdict-ModernBERT-151M",
    execution_backend: activeBackend,
    timestamp: new Date().toISOString(),
    query_id: id || "web_query",
    input_token_count: tokenCount,
    truncated: isTruncated,
    candidate_count: numCandidates,
    calibration: {
      status: isCalibratedScope ? "calibrated_for_scope" : "unvalidated_scope",
      temperature: calibratorTemp,
      method: "L-BFGS scalar temperature scaling"
    },
    temperature_applied: calibratorTemp,
    raw_logits: candidateLogits,
    [probField]: Object.fromEntries(
      candidates.map((c, i) => [c.id, Number(probs[i].toFixed(5))])
    ),
    selected: {
      id: selectedCandidate.id,
      probability: Number(selectedProb.toFixed(5)),
      is_abstention: isAbstention
    },
    policy: {
      threshold: threshold,
      is_autonomous: isAutonomous,
      action: policyAction
    },
    timings_ms: {
      tokenization: Number(tokenizationMs.toFixed(2)),
      forward_readback: Number(inferenceMs.toFixed(2)),
      postprocessing: Number(postprocessingMs.toFixed(2)),
      total_end_to_end: Number(totalMs.toFixed(2))
    }
  };

  self.postMessage({
    type: "RESULT",
    id,
    candidates: candidates.map((c, i) => ({
      ...c,
      prob: probs[i],
      rawLogit: candidateLogits[i]
    })),
    selected: selectedCandidate,
    selectedProb,
    isAbstention,
    isAutonomous,
    policyAction,
    timings: receipt.timings_ms,
    receipt
  });
}

// Module-level serialized queue for message handling
self.onmessage = async function (e) {
  const { type, payload } = e.data;
  if (type === "INIT") {
    if (initPromise) {
      try {
        await initPromise;
      } catch (err) {}
    }
    initPromise = initEngine(payload?.basePath || "artifacts_v2");
    await initPromise;
    initPromise = null;
  } else if (type === "INFER") {
    inferenceQueue = inferenceQueue.then(async () => {
      try {
        await runInference(payload);
      } catch (err) {
        self.postMessage({
          type: "INFER_ERROR",
          id: payload.id,
          message: err.message || String(err)
        });
      }
    });
  }
};
