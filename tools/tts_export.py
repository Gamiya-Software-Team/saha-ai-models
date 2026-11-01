"""Find, export (ONNX) and check offline Sinhala TTS models (runs in GitHub Actions).

  python tools/tts_export.py discover
  python tools/tts_export.py export facebook/mms-tts-sin out/

Quality check: synthesize Sinhala sentences with the exported ONNX model, then
transcribe them with the Sinhala whisper model and report the character error
rate (CER). Lower CER = clearer speech.
"""
import json, os, sys, time, unicodedata, re

SENTENCES = [
    "ඔව්, කියන්න. මම අහගෙන ඉන්නේ.",
    "හරි, වට්ස්ඇප් විවෘත කරනවා.",
    "දැන් වෙලාව උදේ දහයයි තිහයි.",
    "ඔයාගේ බැටරි එකේ චාජ් එක සියයට හැත්තෑපහයි.",
    "ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ.",
    "හෙට උදේ හයට එලාම් එකක් දාන්නද?",
]


def discover():
    from huggingface_hub import HfApi
    api = HfApi()
    seen, rows = set(), []
    for q in ["sinhala tts", "tts sinhala", "mms-tts-sin", "sinhala vits", "sinhala piper", "si_LK", "sinhala speech synthesis"]:
        for m in api.list_models(search=q, sort="downloads", limit=30):
            if m.id not in seen:
                seen.add(m.id); rows.append((m.downloads or 0, m.id, m.pipeline_tag))
    rows.sort(reverse=True)
    print("== Sinhala TTS candidates (downloads, id, task) ==")
    for d, i, t in rows[:40]:
        print(f"{d:>8}  {i}  [{t}]")


def norm(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[^\w\s඀-෿‍]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def export(model_id, out_dir):
    import numpy as np, torch, soundfile as sf
    from transformers import VitsModel, AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).eval()
    sr = model.config.sampling_rate
    print("sampling_rate", sr, "is_uroman", getattr(tok, "is_uroman", None),
          "add_blank", getattr(tok, "add_blank", None), "normalize", getattr(tok, "normalize", None))

    # 1) reference synthesis with PyTorch
    for i, s in enumerate(SENTENCES[:2]):
        with torch.no_grad():
            wav = model(**tok(s, return_tensors="pt")).waveform[0].numpy()
        sf.write(f"{out_dir}/torch_{i}.wav", wav, sr)

    # 2) ONNX export (noise scales as inputs so the app can tune expressiveness)
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids, noise_scale, length_scale, noise_scale_duration):
            self.m.noise_scale = noise_scale
            self.m.speaking_rate = 1.0 / length_scale
            self.m.noise_scale_duration = noise_scale_duration
            return self.m(input_ids=input_ids).waveform

    # VitsModel reads these as python floats; export with fixed defaults instead.
    class Fixed(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids):
            return self.m(input_ids=input_ids).waveform

    model.noise_scale = 0.667
    model.noise_scale_duration = 0.8
    model.speaking_rate = 1.0
    ids = tok(SENTENCES[0], return_tensors="pt")["input_ids"]
    onnx_path = f"{out_dir}/model.onnx"
    torch.onnx.export(Fixed(model), (ids,), onnx_path, input_names=["input_ids"], output_names=["waveform"],
                      dynamic_axes={"input_ids": {1: "n"}, "waveform": {1: "t"}}, opset_version=17, dynamo=False)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, f"{out_dir}/model.int8.onnx", weight_type=QuantType.QUInt8)
    except Exception as e:
        print("int8 quantization failed:", e)

    # 3) tokenizer description for the Kotlin side
    vocab = tok.get_vocab()
    meta = {
        "model": model_id, "sample_rate": sr, "add_blank": bool(getattr(tok, "add_blank", True)),
        "pad_id": int(tok.pad_token_id if tok.pad_token_id is not None else 0),
        "normalize_lowercase": bool(getattr(tok, "normalize", True)),
        "is_uroman": bool(getattr(tok, "is_uroman", False)), "vocab": vocab,
    }
    json.dump(meta, open(f"{out_dir}/tokens.json", "w", encoding="utf-8"), ensure_ascii=False)

    # 4) verify: our own tokenizer re-implementation == HF tokenizer
    def my_ids(text):
        t = text.lower() if meta["normalize_lowercase"] else text
        ids = [vocab[c] for c in t if c in vocab]
        if meta["add_blank"]:
            out = [meta["pad_id"]] * (len(ids) * 2 + 1)
            out[1::2] = ids
            ids = out
        return ids
    mism = 0
    for s in SENTENCES:
        if my_ids(s) != tok(s)["input_ids"]:
            mism += 1
            print("TOKENIZER MISMATCH:", s, my_ids(s)[:20], tok(s)["input_ids"][:20])
    print(f"tokenizer check: {len(SENTENCES)-mism}/{len(SENTENCES)} identical")

    # 5) ONNX synthesis + ASR round trip
    import onnxruntime as ort
    for variant in ["model.onnx", "model.int8.onnx"]:
        p = f"{out_dir}/{variant}"
        if not os.path.exists(p):
            continue
        sess = ort.InferenceSession(p)
        roundtrip(f"vits:{model_id}:{variant}",
                  lambda s: sess.run(None, {"input_ids": np.array([my_ids(s)], dtype=np.int64)})[0][0],
                  sr, out_dir, os.path.getsize(p) / 1e6)


def roundtrip(label, synth, sr, out_dir, size_mb):
    """synth(text) -> float waveform. Prints round-trip CER via Sinhala ASR."""
    import numpy as np, soundfile as sf, librosa, jiwer
    from transformers import pipeline
    global _ASR
    try:
        _ASR
    except NameError:
        _ASR = pipeline("automatic-speech-recognition", model="SinhaSpeech/whisper-small-sinhala-run12-best", device="cpu")
    refs, hyps, rtf = [], [], []
    safe = label.replace("/", "_")
    for i, s in enumerate(SENTENCES):
        t0 = time.time()
        wav = np.asarray(synth(s), dtype="float32").reshape(-1)
        dt = time.time() - t0
        rtf.append(dt / max(0.01, len(wav) / sr))
        sf.write(f"{out_dir}/{safe}_{i}.wav", wav, sr)
        w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        h = _ASR({"raw": w16.astype("float32"), "sampling_rate": 16000},
                 generate_kwargs={"language": "sinhala", "task": "transcribe"})["text"]
        refs.append(norm(s)); hyps.append(norm(h))
        print(f"[{label}] TTS: {s}\n[{label}] ASR: {h}")
    cer = jiwer.cer(refs, hyps) * 100
    print(f"RESULT {label}: size {size_mb:.0f}MB  round-trip CER {cer:.1f}%  RTF {np.mean(rtf):.3f}", flush=True)


def piper(model_id, out_dir):
    """Piper (VITS ONNX) model. Works on-device only if phoneme_type is text (no espeak needed)."""
    import numpy as np, onnxruntime as ort
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(model_id)
    onnx_f = [f for f in files if f.endswith(".onnx")]
    cfg_f = [f for f in files if f.endswith(".onnx.json") or f.endswith("config.json")]
    print(model_id, "files:", files[:30])
    if not onnx_f or not cfg_f:
        print(f"RESULT {model_id}: not a piper model"); return
    cfg = json.load(open(hf_hub_download(model_id, cfg_f[0]), encoding="utf-8"))
    ptype = cfg.get("phoneme_type", "espeak")
    print(model_id, "phoneme_type:", ptype, "espeak voice:", cfg.get("espeak", {}), "sr:", cfg.get("audio", {}).get("sample_rate"))
    pmap = cfg["phoneme_id_map"]
    sr = cfg["audio"]["sample_rate"]
    inf = cfg.get("inference", {})
    scales = np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0), inf.get("noise_w", 0.8)], dtype=np.float32)
    path = hf_hub_download(model_id, onnx_f[0])
    sess = ort.InferenceSession(path)

    def phonemes(text):
        if ptype == "text":
            return list(unicodedata.normalize("NFD", text))
        try:
            from piper_phonemize import phonemize_espeak
            voice = cfg.get("espeak", {}).get("voice", "si")
            return [p for sent in phonemize_espeak(text, voice) for p in sent]
        except Exception as e:
            raise RuntimeError(f"espeak phonemizer needed: {e}")

    def synth(text):
        ids = list(pmap["^"])
        for ph in phonemes(text):
            if ph in pmap:
                ids += pmap[ph] + pmap["_"]
        ids += pmap["$"]
        x = np.array([ids], dtype=np.int64)
        feeds = {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64), "scales": scales}
        if "sid" in [i.name for i in sess.get_inputs()]:
            feeds["sid"] = np.array([0], dtype=np.int64)
        return sess.run(None, feeds)[0]

    try:
        roundtrip(f"piper:{model_id}[{ptype}]", synth, sr, out_dir, os.path.getsize(path) / 1e6)
    except Exception as e:
        print(f"RESULT {model_id}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    if sys.argv[1] == "discover":
        discover()
    elif sys.argv[1] == "piper":
        piper(sys.argv[2], sys.argv[3])
    else:
        export(sys.argv[2], sys.argv[3])
