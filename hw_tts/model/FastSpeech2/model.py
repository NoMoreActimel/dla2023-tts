import torch

from torch import nn

from .encoder_decoder import Encoder, Decoder
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, config):
        """
            FastSpeech2 model implementation
            from https://arxiv.org/pdf/2006.04558.pdf

            This implementation may vary slightly from the original paper,
            as there are 2 versions of the paper and some people
            proposed somewhat different approaches in terms of
            pitch and energy data handling
        """
        super().__init__()


        self.model_config = config["model"]
        self.preprocessing_config = config["data_preprocessing"]
        
        self.encoder = Encoder(self.model_config)

        max_decoder_len = self.model_config.get("max_mel_length", None)
        self.decoder = Decoder(self.model_config, len_max_seq=max_decoder_len)

        self.variance_adaptor = VarianceAdaptor(config=config, model_config=self.model_config)

    def forward(
            self,
            src_seq,
            src_pos,
            mel_pos,
            max_mel_length,
            duration_target=None,
            pitch_target=None,
            energy_target=None,
            duration_coeff=1.0,
            pitch_coeff=1.0,
            energy_coeff=1.0
    ):
        output = self.encoder(src_seq, src_pos)
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

        max_length_mask = torch.arange(0, max_mel_length).unsqueeze(0).to(output.device)
        mel_mask = (max_length_mask >= mel_pos).unsqueeze(1, 2)
        mel_mask = mel_mask.expand(-1, -1, output.shape[2])
        output = output.masked_fill(mel_mask, 0.0)

        output = self.mel_linear(output)

        return {
            "mel-spectrogram": output,
            "duration": adaptor_output["duration"],
            "pitch": adaptor_output["pitch"],
            "energy": adaptor_output["energy"],
            "mel-length": adaptor_output["mel-length"]
        }
