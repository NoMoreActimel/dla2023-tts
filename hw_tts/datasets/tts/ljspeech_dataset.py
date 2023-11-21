import json
import librosa
import logging
import numpy as np
import os
import shutil
from pathlib import Path
import torchaudio

from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_tts.base.base_dataset import BaseDataset
from hw_tts.utils import ROOT_PATH
from hw_tts.datasets.tts.ljspeech_preprocessor import LJSpeechPreprocessor

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

class LJspeechFastSpeech2Dataset(BaseDataset):
    def __init__(self, config=None, raw_data_dir=None, data_dir=None, *args, **kwargs):
        if config is None:
            config = kwargs["config_parser"]

        if data_dir is None:
            data_dir = config["preprocessing"].get("data_dir", None)
            if data_dir is None:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech_processed"
            else:
                data_dir = Path(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
        if raw_data_dir is None:
            raw_data_dir = config["preprocessing"].get("raw_data_dir", None)
            if raw_data_dir is None:
                raw_data_dir = ROOT_PATH / "data" / "ljspeech"
            else:
                raw_data_dir = Path(raw_data_dir)
            raw_data_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = config

        self._raw_data_dir = raw_data_dir
        self._data_dir = data_dir

        self.max_wav_value = self.config["preprocessing"]["max_wav_value"]
        self.sample_rate = self.config["preprocessing"]["sr"]

        self._load_dataset()

        self.data_processor = LJSpeechPreprocessor(raw_data_dir, data_dir, self.config_parser)
        self.data_processor.process()
    
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

        # files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        # train_length = int(0.85 * len(files)) # hand split, test ~ 15% 
        # (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        # (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        # for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
        #     if i < train_length:
        #         shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))
        #     else:
        #         shutil.move(str(fpath), str(self._data_dir / "test" / fpath.name))
        # shutil.rmtree(str(self._data_dir / "wavs"))

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
        split_dir = self._data_dir
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists():
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index
    