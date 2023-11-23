import torch

from pathlib import Path

from utils import get_WaveGlow
import hw_tts.waveglow as waveglow


def move_batch_to_device(batch, device: torch.device):
    """
    Move all necessary tensors to the HPU
    """
    for tensor_for_gpu in [
        "src_seq", "src_pos", "mel_target", "mel_pos", 
        "duration_target", "pitch_target", "energy_target"
    ]:
        batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    
    return batch


def run_inference(
        model,
        dataset,
        indices,
        waveglow_path,
        dataset_type="train",
        inference_path="",
        duration_coeffs=[1.0],
        pitch_coeffs=[1.0],
        energy_coeffs=[1.0],
        epoch=None
    ):
    path = Path(inference_path)
    path.mkdir(exist_ok=True, parents=True)

    WaveGlow = get_WaveGlow(waveglow_path)

    batch = dataset.collate_fn([dataset[ind] for ind in indices])
    batch = move_batch_to_device(batch, device='cuda:0')

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
                        f"/{dataset_type}_epoch{epoch}_utterance_{ind}:_" \
                        f"duration={duration_coeff}_pitch={pitch_coeff}_" \
                        f"energy={energy_coeff}.wav"
                    waveglow.inference(mel_predict.unsqueeze(0), WaveGlow, path)
