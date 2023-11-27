import torch
import numpy as np

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
    inference_path = Path(inference_path) / dataset_type
    inference_path.mkdir(exist_ok=True, parents=True)

    inference_paths = [inference_path / f"utterance_{ind}" for ind in indices]
    
    for i, path in enumerate(inference_paths):
        if epoch is not None:
            inference_paths[i] = path / f"epoch{epoch}"
        inference_paths[i].mkdir(exist_ok=True, parents=True)

    WaveGlow = get_WaveGlow(waveglow_path)

    dataset_items = [dataset[ind] for ind in indices]
    batch = dataset.collate_fn(dataset_items)
    batch = move_batch_to_device(batch, device='cuda:0')

    paths = []

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

                for i, (ind, mel_predict) in enumerate(zip(indices, mel_predicts)):
                    filename =  f"duration={duration_coeff}_pitch={pitch_coeff}_" \
                                f"energy={energy_coeff}"

                    np.save(inference_paths[i] / (filename + ".spec"), mel_predict.cpu())
                    waveglow.inference(
                        mel_predict.unsqueeze(0),
                        WaveGlow,
                        inference_paths[i] / (filename + ".wav")
                    )
                    paths.append(path)
    
    return paths
