"""Debug helpers: VITS tokenizer behaviour + espeak-ng Sinhala phonemes."""
import json, sys
from tts_export import SENTENCES, roundtrip, norm

def vits_hf(model_id, out):
    import os, torch
    from transformers import VitsModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id); model = VitsModel.from_pretrained(model_id).eval()
    v = tok.get_vocab()
    print("vocab size", len(v), "sample", sorted(v.items(), key=lambda x: x[1])[:60])
    for s in SENTENCES[:2]:
        print("HF ids", tok(s)["input_ids"][:40], tok.convert_ids_to_tokens(tok(s)["input_ids"][:40]))
    os.makedirs(out, exist_ok=True)
    def synth(s):
        with torch.no_grad():
            return model(**tok(s, return_tensors="pt")).waveform[0].numpy()
    roundtrip(f"vits-torch-hftok:{model_id}", synth, model.config.sampling_rate, out, 0)

def espeak():
    from piper_phonemize import phonemize_espeak
    import piper_phonemize, os
    print("piper_phonemize at", os.path.dirname(piper_phonemize.__file__))
    for s in SENTENCES + ["වට්සැප්", "කැමරාව", "අම්මාට call කරන්නද?", "සියයට පනහයි"]:
        print(s, "=>", "|".join("".join(x) for x in phonemize_espeak(s, "si")))

if __name__ == "__main__":
    if sys.argv[1] == "vits":
        vits_hf(sys.argv[2], sys.argv[3])
    else:
        espeak()
