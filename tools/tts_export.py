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
    from transformers import pipeline
    import librosa, jiwer
    asr = pipeline("automatic-speech-recognition", model="SinhaSpeech/whisper-small-sinhala-run12-best", device="cpu")
    for variant in ["model.onnx", "model.int8.onnx"]:
        p = f"{out_dir}/{variant}"
        if not os.path.exists(p):
            continue
        sess = ort.InferenceSession(p)
        refs, hyps, rtf = [], [], []
        for i, s in enumerate(SENTENCES):
            x = np.array([my_ids(s)], dtype=np.int64)
            t0 = time.time()
            wav = sess.run(None, {"input_ids": x})[0][0]
            dt = time.time() - t0
            rtf.append(dt / (len(wav) / sr))
            sf.write(f"{out_dir}/{variant}_{i}.wav", wav, sr)
            w16 = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000) if sr != 16000 else wav
            h = asr({"raw": w16.astype("float32"), "sampling_rate": 16000},
                    generate_kwargs={"language": "sinhala", "task": "transcribe"})["text"]
            refs.append(norm(s)); hyps.append(norm(h))
            print(f"[{variant}] TTS: {s}\n[{variant}] ASR: {h}")
        cer = jiwer.cer(refs, hyps) * 100
        print(f"RESULT {model_id} {variant}: size {os.path.getsize(p)/1e6:.0f}MB  round-trip CER {cer:.1f}%  "
              f"RTF {np.mean(rtf):.3f} (runner CPU)")


if __name__ == "__main__":
    if sys.argv[1] == "discover":
        discover()
    else:
        export(sys.argv[2], sys.argv[3])
