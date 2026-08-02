#!/usr/bin/env python3
"""
Nyawa BGE Embedder Server
Uses ONNX Runtime + tokenizers for local embedding inference.
No PyTorch, no GPU, no CGO. Pure Python + ONNX.

Protocol: JSON-RPC over stdin/stdout
  Request:  {"jsonrpc":"2.0","id":1,"method":"embed","params":{"text":"..."}}
  Response: {"jsonrpc":"2.0","id":1,"result":{"embedding":[0.1,0.2,...],"dim":384}}
"""
import json
import sys
import os
import time
import numpy as np
import onnxruntime

# Try to import tokenizers (much lighter than transformers)
try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

MODEL_DIR = os.environ.get("NYAWA_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))

class BgeEmbedder:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.session = None
        self.tokenizer = None
        self.dim = 384
        self._load()

    def _load(self):
        model_path = os.path.join(self.model_dir, "model.onnx")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise RuntimeError(f"Tokenizer not found: {tokenizer_path}")

        # Load ONNX model
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(model_path, opts)

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self.tokenizer.enable_truncation(max_length=128)

        # Get model input names
        input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_name = input_names[0]

        # Get output dimension
        self.dim = self.session.get_outputs()[0].shape[-1] or 384
        print(f"Model loaded: {os.path.basename(model_path)} dim={self.dim} inputs={input_names}", file=sys.stderr)

    def embed(self, text):
        """Compute embedding for a single text."""
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

        # Determine model inputs
        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        has_token_type = any("token_type" in name or "segment" in name for name in [inp.name for inp in self.session.get_inputs()])
        if has_token_type:
            onnx_inputs["token_type_ids"] = token_type_ids

        # Run model
        outputs = self.session.run(None, onnx_inputs)

        # Mean pooling
        embedding = outputs[0]  # (1, seq_len, hidden)
        mask = attention_mask.astype(np.float32)  # (1, seq_len)
        mask = np.expand_dims(mask, axis=-1)  # (1, seq_len, 1)
        embedding = (embedding * mask).sum(axis=1)  # (1, hidden)
        mask_sum = mask.sum(axis=1).clip(min=1e-9)  # (1, 1)
        embedding = embedding / mask_sum
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding[0].tolist()


def main():
    # Read model path from args or env
    model_dir = os.environ.get("NYAWA_MODEL_DIR", None)
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        model_dir = sys.argv[1]

    if model_dir:
        # Override MODEL_DIR from arg
        global MODEL_DIR
        MODEL_DIR = model_dir

    embedder = BgeEmbedder(model_dir=os.environ.get("NYAWA_MODEL_DIR", MODEL_DIR))
    
    # Write readiness marker
    print("READY", file=sys.stderr, flush=True)

    # JSON-RPC loop over stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "embed":
            text = params.get("text", "")
            try:
                emb = embedder.embed(text)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"embedding": emb, "dim": embedder.dim},
                }
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -1, "message": str(e)},
                }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
