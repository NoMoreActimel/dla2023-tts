import torch

from utils import get_WaveGlow
import hw_tts.waveglow as waveglow

def run_inference(
        model,
        dataset,
        indices,
        dataset_type="train",
        inference_path="",
        duration_coeffs=[1.0],
        pitch_coeffs=[1.0],
        energy_coeffs=[1.0]
    ):

    WaveGlow = get_WaveGlow()

    batch = dataset.collate_fn([dataset[ind] for ind in indices])

    for duration_coeff in duration_coeffs:
        for pitch_coeff in pitch_coeffs:
            for energy_coeff in energy_coeffs:
                with torch.no_grad():
                    output = model.forward(**{
                        "src_seq": batch["src_seq"],
                        "src_pos": batch["src_pos"],
                        "duration_coeff": duration_coeff,
                        "pitch_coeff": pitch_coeff,
                        "energy_coeff": energy_coeff
                    })
                
                mel_predicts = output["mel_predict"].transpose(1, 2)

                for ind, mel_predict in zip(indices, mel_predicts):
                    path = inference_path + \
                        f"{dataset_type}_utterance_{ind}:_" \
                        f"duration={duration_coeff}_pitch={pitch_coeff}_" \
                        f"energy={energy_coeff}"
                    waveglow.inference.inference(mel_predict, WaveGlow, path)