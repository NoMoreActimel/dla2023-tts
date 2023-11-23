import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import librosa
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
from inference import run_inference


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            text_encoder=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(
            model, criterion, metrics,
            optimizer, lr_scheduler, config, device
        )
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", "grad norm", 
            *[m.name for m in self.metrics if self._compute_on_train(m)],
            writer=self.writer
        )

        self.datasets = {
            key: dataloader.dataset for key, dataloader in dataloaders.items()
        }
        self.inference_indices = {
            "train": [30, 40, 50],
            "val": [30, 40, 50]
        }
        self.inference_paths = {
            "train": config["data"]["train"]["inference_path"],
            "val": config["data"]["val"]["inference_path"]
        }
        self.waveglow_paths = {
            "train": config["data"]["train"]["waveglow_path"],
            "val": config["data"]["val"]["waveglow_path"]
        }

        if self.device.type != "cuda":
            print("WaveGlow inference could be processed only on cuda:0, now on cpu!")


    @staticmethod
    def _compute_on_train(metric):
        if hasattr(metric, "compute_on_train"):
            return metric.compute_on_train
        return True

    @staticmethod
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

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics_tracker=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )

                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    last_lr = self.optimizer.param_groups[0]['lr']
                else:
                    last_lr = self.lr_scheduler.get_last_lr()[0]

                self.writer.add_scalar("learning rate", last_lr)
                self._log_spectrogram(batch["mel_predict"].detach().transpose(1, 2))
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break

        if self.device.type == "cuda":
            for dataset_type in ["train", "val"]:
                inference_paths = run_inference(
                    model=self.model,
                    dataset=self.datasets[dataset_type],
                    dataset_type="train",
                    inference_path=self.inference_paths[dataset_type],
                    waveglow_path=self.waveglow_paths[dataset_type],
                    indices=self.inference_indices[dataset_type],
                    duration_coeffs=[1.0],
                    pitch_coeffs=[1.0],
                    energy_coeffs=[1.0],
                    epoch=epoch
                )
                # for inference_path in inference_paths:
                #     self._log_predictions(inference_path, dataset_type)

        log = last_train_metrics
        return log


    def process_batch(
            self, batch, is_train: bool, metrics_tracker: MetricTracker,
            rare_metrics_tracker: MetricTracker = None
            ):
    
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)
        
        losses = self.criterion(**batch)
        loss_names = "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss"
        for i, loss_name in enumerate(loss_names):
                batch[loss_name] = losses[i]

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()
        
        for loss_name in loss_names:
            metrics_tracker.update(loss_name, batch[loss_name].item())

        for met in self.metrics:
            metrics_tracker.update(met.name, met(**batch))
        
        if rare_metrics_tracker:
            for met in self.rare_eval_metrics:
                rare_metrics_tracker.update(met.name, met(**batch))
    
        return batch


    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch, name="spectrogram"):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    def _log_audio(self, audio, name="audio"):
        sample_rate = self.config["preprocessing"]["sr"]
        self.writer.add_audio(name, audio, sample_rate=sample_rate)
    
    def _log_predictions(self, paths, dataset_type):
        # not tested yet
        ind = random.choice(self.inference_indices[dataset_type])
        path = paths[ind]
        name = f"utterance_{ind}"
        
        wav, _ = librosa.load(path + ".wav")
        self._log_audio(wav, "audio_" + name)

        spec = np.load(path + ".spec")
        self._log_spectrogram([spec], "spectrogram_" + name)


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
