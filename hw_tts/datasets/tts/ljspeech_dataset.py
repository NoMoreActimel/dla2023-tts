import json
import librosa
import logging
import numpy as np
import os
import shutil
from pathlib import Path
import torch
import torchaudio

from speechbrain.utils.data_utils import download_file
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from hw_tts.base.base_dataset import BaseDataset
from hw_tts.datasets.tts.ljspeech_preprocessor import LJSpeechPreprocessor
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.text import text_to_sequence

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

class LJspeechFastSpeech2Dataset(BaseDataset):
    def __init__(self, config=None, raw_data_dir=None, data_dir=None, *args, **kwargs):
        if config is None:
            config = kwargs["config_parser"]
        self.config = config

        if data_dir is None:
            data_dir = config["preprocessing"].get("data_dir", None)
            if data_dir is None:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech_processed"
        if raw_data_dir is None:
            raw_data_dir = config["preprocessing"].get("raw_data_dir", None)
            if raw_data_dir is None:
                raw_data_dir = ROOT_PATH / "data" / "ljspeech"

        self._raw_data_dir = Path(raw_data_dir)
        self._raw_data_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(exist_ok=True, parents=True)

        self.max_wav_value = self.config["preprocessing"]["max_wav_value"]
        self.sample_rate = self.config["preprocessing"]["sr"]

        if len(os.listdir(self._raw_data_dir)) == 0:
            self._load_dataset()

        self.data_processor = LJSpeechPreprocessor(self._raw_data_dir, self._data_dir, self.config)
        if self.config["preprocessing"].get("perform_preprocessing", None):
            self.data_processor.process()
            
        self.spec_dir = self.data_processor.spec_path
        self.duration_dir = self.data_processor.duration_path
        self.pitch_dir = self.data_processor.pitch_path
        self.energy_dir = self.data_processor.energy_path
    
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._raw_data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._raw_data_dir)
        for fpath in (self._raw_data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._raw_data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._raw_data_dir / "LJSpeech-1.1"))
        print(f"Unpacked LJSpeech to {self._raw_data_dir}")

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        name = data_dict["name"]
        raw_text = data_dict["raw_text"]
        text = np.array(text_to_sequence(data_dict["text"], ["english_cleaners"]))
        spectrogram = np.load(self.spec_dir / f"{name}_spec.npy")
        duration = np.load(self.duration_dir / f"{name}_duration.npy")
        pitch = np.load(self.pitch_dir / f"{name}_pitch.npy")
        energy = np.load(self.energy_dir / f"{name}_energy.npy")

        return {
            "name": name,
            "raw_text": raw_text,
            "text": torch.from_numpy(text).long(),
            "spectrogram": torch.from_numpy(spectrogram).float(),
            "duration": torch.from_numpy(duration).long(),
            "pitch": torch.from_numpy(pitch).float(),
            "energy": torch.from_numpy(energy).float()
        }

    def _get_or_load_index(self):
        index_path = self._data_dir / f"index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []

        train_metadata = self.data_processor.train_metadata_filename
        val_metadata = self.data_processor.val_metadata_filename

        for metadata_filename in [train_metadata, val_metadata]:
            metadata_path = self._data_dir / metadata_filename
            index = []
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    name, text, raw_text = line.strip('\n').split('\t')
                    index.append({"name": name, "text": text,"raw_text": raw_text})
        
        return index
    
    @classmethod
    def collate_fn(dataset_items):
        stats = {
            "name": [], "raw_text": [], "text": [], "text_length": [],
            "spectrogram": [], "spectrogram_length": [],
            "duration": [], "pitch": [], "energy": []
        }

        for item in dataset_items:
            for key in stats.keys():
                if key in ["text_length", "spectrogram_length"]:
                    stats[key].append(item[key.split('_')[0]].shape[0])
                else:
                    stats[key].append(item[key])
        
        for key, values in stats.items():
            if key in ["text", "spectrogram", "duration", "pitch", "energy"]:
                stats[key] = pad_sequence(values, batch_first=True)
        
        return stats


