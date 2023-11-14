import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            rare_eval_metrics,
            n_epochs_frequency,
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
            model, criterion, metrics, rare_eval_metrics, n_epochs_frequency,
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
            "loss", "sisdr_loss", "ce_loss", "speaker_accuracy", "grad norm", 
            *[m.name for m in self.metrics if self._compute_on_train(m)],
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "sisdr_loss", "ce_loss", "speaker_accuracy",
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.rare_evaluation_metrics = MetricTracker(
            *[m.name for m in self.rare_eval_metrics], writer=self.writer
        ) if self.n_epochs_frequency else None


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
        for tensor_for_gpu in ["input", "ref", "ref_length", "target", "audio_length", "speaker_id"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch, do_rare_eval=False):
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
                # self._log_predictions(**batch, log_rare_metrics=do_rare_eval)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch["input"], "mixed_audio")
                self._log_audio(batch["ref"], "ref_audio")
                self._log_audio(batch["predicts"]["L1"], "predicted_audio")
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log, rare_val_log = self._evaluation_epoch(epoch, part, dataloader, do_rare_eval)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
            if do_rare_eval:
                log.update(**{f"{part}_{name}": value for name, value in rare_val_log.items()})
        
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(val_log["loss"])

        return log

    def process_batch(
            self, batch, is_train: bool, metrics_tracker: MetricTracker,
            rare_metrics_tracker: MetricTracker = None
            ):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs, speaker_logits = self.model(**batch)
        batch["predicts"] = outputs
        batch["speaker_logits"] = speaker_logits

        speaker_predicts = torch.softmax(batch["speaker_logits"], dim=1).argmax(dim=1)
        speaker_accuracy = (speaker_predicts == batch["speaker_id"]).sum() / speaker_predicts.shape[0]

        # if type(outputs) is dict:
        #     batch.update(outputs)
        # else:
        #     batch["logits"] = outputs

        # batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        # batch["log_probs_length"] = self.model.transform_input_lengths(
        #     batch["spectrogram_length"]
        # )
        batch["loss"], batch["sisdr_loss"], batch["ce_loss"] = self.criterion(eval_mode=(not is_train), **batch)

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

        metrics_tracker.update("loss", batch["loss"].item())
        metrics_tracker.update("sisdr_loss", batch["sisdr_loss"].item())
        metrics_tracker.update("ce_loss", batch["ce_loss"].item())
        metrics_tracker.update("speaker_accuracy", speaker_accuracy.item())

        for met in self.metrics:
            metrics_tracker.update(met.name, met(**batch))
        
        if rare_metrics_tracker:
            for met in self.rare_eval_metrics:
                rare_metrics_tracker.update(met.name, met(**batch))
    
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader, do_rare_eval=False):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        if do_rare_eval and self.rare_evaluation_metrics:
            self.rare_evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics_tracker=self.evaluation_metrics,
                    rare_metrics_tracker=self.rare_evaluation_metrics if do_rare_eval else None
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            # self._log_predictions(**batch, log_rare_metrics=do_rare_eval)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_audio(batch["input"], "mixed_audio")
            self._log_audio(batch["ref"], "ref_audio")
            self._log_audio(batch["predicts"]["L1"], "predicted_audio")

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        
        return self.evaluation_metrics.result(), self.rare_evaluation_metrics.result() if do_rare_eval else None

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            audio_path,
            examples_to_log=10,
            log_rare_metrics=False,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        
        log_probs = log_probs[:examples_to_log].detach().cpu()
        log_probs_length = log_probs_length[:examples_to_log].detach().cpu().numpy()

        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in list(zip(argmax_inds, log_probs_length))
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]

        if hasattr(self.text_encoder, "ctc_decode"):
            argmax_texts = [
                self.text_encoder.ctc_decode(inds) 
                for inds in argmax_inds
            ]
            if self.text_encoder.use_lm:
                beam_search_predictions = self.text_encoder.ctc_lm_beam_search(
                    log_probs, log_probs_length
                ) if log_rare_metrics else None
            else:
                beam_search_predictions = [
                    self.text_encoder.ctc_beam_search(log_probs_line, length)[0].text
                    for log_probs_line, length in list(zip(log_probs, log_probs_length))
                ] if log_rare_metrics else None
            # here we log predefined metrics, so
            # we hardcoded beamsearch logging, as it was with argmax-texts
            # each new rare-metric logging should be implemented the same way
        
        tuples = list(zip(text, argmax_texts_raw, audio_path))
        shuffle(tuples)
        rows = {}
        for i, (target, raw_pred, audio_path) in enumerate(tuples[:examples_to_log]):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred = argmax_texts[i]
                beam_search_pred = beam_search_predictions[i] if beam_search_predictions else None

            target = BaseTextEncoder.normalize_text(target)
            wer_raw = calc_wer(target, raw_pred) * 100
            cer_raw = calc_cer(target, raw_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "wer raw": wer_raw,
                "cer raw": cer_raw,
            }

            if hasattr(self.text_encoder, "ctc_decode"):
                wer_argmax = calc_wer(target, pred) * 100
                cer_argmax = calc_cer(target, pred) * 100
                rows[Path(audio_path).name].update({
                    "predictions": pred,
                    "beam search prediction": beam_search_pred,
                    "wer argmax": wer_argmax,
                    "cer argmax": cer_argmax,
                })
                
                if log_rare_metrics:
                    wer_beamsearch = calc_wer(target, beam_search_pred) * 100
                    cer_beamsearch = calc_cer(target, beam_search_pred) * 100
                    rows[Path(audio_path).name].update({
                        "wer beamsearch": wer_beamsearch,
                        "cer beamsearch": cer_beamsearch
                    })

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio_batch, audio_name="audio"):
        audio = random.choice(audio_batch)
        sample_rate = self.config["preprocessing"]["sr"]
        self.writer.add_audio(audio_name, audio, sample_rate=sample_rate)

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
