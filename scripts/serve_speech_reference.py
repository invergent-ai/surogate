#!/usr/bin/env python3
"""Generate independent NeMo oracle tensors for the native speech test.

Run in an environment with NeMo ASR, soundfile, and safetensors installed:
  python scripts/serve_speech_reference.py MODEL.nemo LM.nemo audio.wav output.safetensors
  test_speech_model PREPARED_DIR output.safetensors [cpu|GPU_INDEX]

Nothing from NeMo is needed by the native runtime or normal model preparation.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("lm")
    parser.add_argument("audio")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = ASRModel.restore_from(args.model, map_location=args.device).eval()
    samples, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or samples.ndim != 1 or len(samples) <= 256:
        raise ValueError("oracle audio must be mono 16 kHz and longer than 256 samples")
    model.change_decoding_strategy(
        OmegaConf.create({"strategy": "greedy_batch", "greedy": {"use_cuda_graph_decoder": False}}), decoder_type="rnnt"
    )
    streaming = model.encoder.att_context_style == "chunked_limited"
    model.change_decoding_strategy(
        OmegaConf.create(
            {
                "strategy": "beam_batch",
                "beam": {
                    "beam_size": 64 if streaming else 32,
                    "ngram_lm_model": args.lm,
                    "ngram_lm_alpha": 0.55 if streaming else 0.5,
                    "beam_beta": 1.75 if streaming else 2.0,
                    "allow_cuda_graphs": False,
                },
            }
        ),
        decoder_type="ctc",
    )
    audio = torch.tensor(samples, device=args.device)[None]
    length = torch.tensor([len(samples)], device=args.device)
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=audio, length=length)
        context = model.encoder.att_context_size
        model.encoder.set_default_att_context_size([-1, -1])
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        ctc = model.ctc_decoder(encoder_output=encoded)
        decoded = model.ctc_decoding.ctc_decoder_predictions_tensor(ctc, encoded_len)
        final = decoded[0].text if hasattr(decoded[0], "text") else decoded[0]
        model.encoder.set_default_att_context_size(context)
        tensors = {
            "pcm": audio,
            "mel": mel[..., : int(mel_len[0])],
            "encoded": encoded[..., : int(encoded_len[0])],
            "ctc": ctc[:, : int(encoded_len[0])],
        }
        tdt = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )[0].text
        partials = []
        if streaming:
            # NeMo's feature transform with zero-padded streaming boundaries.
            f = model.preprocessor.featurizer
            x = audio[0]
            emphasized = torch.cat((x[:1], x[1:] - f.preemph * x[:-1]))
            spectrum = torch.stft(
                torch.nn.functional.pad(emphasized, (256, 256)),
                n_fft=f.n_fft,
                hop_length=f.hop_length,
                win_length=f.win_length,
                window=f.window,
                center=False,
                return_complex=True,
            )
            magnitude = torch.view_as_real(spectrum).pow(2).sum(-1).sqrt().pow(f.mag_power)
            features = torch.log(f.fb @ magnitude + f.log_zero_guard_value_fn(magnitude))[..., : len(samples) // 160]
            tensors["stream_mel"] = features
            cache = model.encoder.get_initial_cache_state(batch_size=1)
            history = features[..., :0]
            offset = 0
            hypothesis = None
            partials = []
            while offset < features.shape[-1]:
                step = len(partials)
                cfg = model.encoder.streaming_cfg

                def value(name, cfg=cfg, step=step):
                    setting = getattr(cfg, name)
                    return setting[bool(step)] if isinstance(setting, (list, tuple)) else setting

                chunk = features[..., offset : offset + value("chunk_size")]
                context_len = value("pre_encode_cache_size")
                previous = history[..., -context_len:] if context_len else history[..., :0]
                signal = torch.cat((previous, chunk), -1)
                result = model.encoder.cache_aware_stream_step(
                    processed_signal=signal,
                    processed_signal_length=torch.tensor([signal.shape[-1]], device=args.device),
                    cache_last_channel=cache[0],
                    cache_last_time=cache[1],
                    cache_last_channel_len=cache[2],
                    drop_extra_pre_encoded=cfg.drop_extra_pre_encoded if step else 0,
                    keep_all_outputs=offset + chunk.shape[-1] == features.shape[-1],
                )
                enc, enc_len, *cache = result
                hypothesis = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True, partial_hypotheses=hypothesis
                )
                tensors[f"chunk_{step}"] = signal
                tensors[f"chunk_encoded_{step}"] = enc[..., : int(enc_len[0])]
                partials.append(hypothesis[0].text)
                history = torch.cat((history, chunk), -1)[..., -cfg.pre_encode_cache_size[-1] :]
                offset += chunk.shape[-1]
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"partials": partials, "final": final, "tdt": tdt, "streaming": streaming}, ensure_ascii=False)
    )
    print(json.dumps({"chunks": len(partials), "final": final}, ensure_ascii=False))


if __name__ == "__main__":
    main()
