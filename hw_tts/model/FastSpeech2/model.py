import torch
# import waveglow

from torch import nn

from .encoder_decoder import Encoder, Decoder
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, config, WaveGlow=None):
        """
            FastSpeech2 model implementation
            from https://arxiv.org/pdf/2006.04558.pdf

            This implementation may vary slightly from the original paper,
            as there are 2 versions of the paper and some people
            proposed somewhat different approaches in terms of
            pitch and energy data handling

            WaveGlow: WaveGlow object for inference
        """
        super().__init__()

        self.model_config = config["model"]["args"]
        self.preprocessing_config = config["preprocessing"]
        self.WaveGlow = WaveGlow
        
        self.encoder = Encoder(self.model_config["encoder_decoder"])
        self.decoder = Decoder(self.model_config["encoder_decoder"])

        self.variance_adaptor = VarianceAdaptor(self.model_config)

        self.mel_decoder = nn.Linear(
            self.model_config["encoder_decoder"]["decoder_dim"],
            self.model_config["n_mel_channels"]
        )

    def forward(
            self,
            src_seq,
            src_pos,
            mel_pos=None,
            max_mel_length=None,
            duration_target=None,
            pitch_target=None,
            energy_target=None,
            duration_coeff=1.0,
            pitch_coeff=1.0,
            energy_coeff=1.0,
            **kwargs
    ):
        output, non_pad_mask = self.encoder(src_seq, src_pos)
        adaptor_output = self.variance_adaptor(
            output,
            max_mel_length,
            duration_target,
            pitch_target,
            energy_target,
            duration_coeff,
            pitch_coeff,
            energy_coeff
        )

        output = adaptor_output["mel-spectrogram"]
        output = self.decoder(output, mel_pos)

        mel_mask = self.get_mel_mask(output, mel_pos, max_mel_length)
        # max_length_mask = torch.arange(0, max_mel_length).unsqueeze(0).to(output.device)
        # mel_mask = (max_length_mask >= mel_pos).unsqueeze(1, 2)
        # mel_mask = mel_mask.expand(-1, -1, output.shape[2])
        output = output.masked_fill(mel_mask, 0.0)

        output = self.mel_decoder(output)

        return {
            "mel_predict": output,
            "log_duration_predict": adaptor_output["log_duration"],
            "log_pitch_predict": adaptor_output["pitch"],
            "log_energy_predict": adaptor_output["energy"],
            "mel_length_predict": adaptor_output["mel-length"]
        }
    
    def get_mel_mask(self, mel, mel_pos, max_mel_length):
        if mel_pos is None:
            return None

        length = torch.max(mel_pos, dim=-1)
        if max_mel_length is None:
            max_mel_length = torch.max(length).item()
        
        ids = torch.arange(0, max_mel_length).unsqueeze(0).to(mel.device)
        mask = (ids >= mel_pos).bool().unsqueeze(-1)

        return mask
        

    def inference(
            self,
            src_seq,
            src_pos,
            duration_coeff=1.0,
            pitch_coeff=1.0,
            energy_coeff=1.0,
            WaveGlow=None):
    
        if WaveGlow is not None:
            if self.WaveGlow is None:
                self.WaveGlow = WaveGlow
        else:
            assert self.WaveGlow is not None, f"No WaveGlow provided for inference"
            WaveGlow = self.WaveGlow
        
        with torch.no_grad():
            output = self.forward(
                src_seq,
                src_pos,
                duration_coeff=duration_coeff,
                pitch_coeff=pitch_coeff,
                energy_coeff=energy_coeff
            )
        
        mel_predict = output["mel_predict"].transpose(1, 2)
        desc = f"predicts for: duration = {duration_coeff}, "\
            f"pitch = {pitch_coeff}, energy = {energy_coeff}"
        waveglow.inference.inference(mel_predict, WaveGlow, desc)
