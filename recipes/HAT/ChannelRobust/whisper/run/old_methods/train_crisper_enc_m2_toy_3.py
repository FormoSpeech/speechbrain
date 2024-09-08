import logging
import os
import sys
from pathlib import Path
import time

import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.core import AMPConfig

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm.contrib import tqdm
from speechbrain.dataio.dataloader import LoopedLoader, SaveableDataLoader

import wandb

logger = logging.getLogger(__name__)

# Define training procedure
class Crisper(sb.Brain):
    
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        channel_ids, _ = batch.channel_id_encoded
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        stem_ids, _ = batch.stem_encoded

        total_loss = 0
        num_comparisons = 0

        # Iterate over the batch, since each group of 2 corresponds to a stem
        for i in range(0, len(channel_ids), 3):
            stem_channels = channel_ids[i:i+3]
            stems = stem_ids[i:i+3]
            
            assert stems[0][0] == stems[1][0]
            
            wavs_stem = wavs[i:i+3]
            bos_tokens_stem = bos_tokens[i:i+3]

            # Identify the index of target channel and the others
            target_index = (stem_channels == 2).nonzero(as_tuple=True)[0].item()
            other_indices = [j for j in range(3) if j != target_index]

            # Process target channel with Whisper
            wavs_target = wavs_stem[target_index].unsqueeze(0)  # Keep it in batch form
            bos_tokens_target = bos_tokens_stem[target_index].unsqueeze(0)
            
            enc_out_ref, _, _ = self.modules.whisper(wavs_target, bos_tokens_target)
            enc_out_ref_last = enc_out_ref[-1]

            # Process other channels with Crisper
            wavs_other = wavs_stem[other_indices]
            bos_tokens_other = bos_tokens_stem[other_indices]
            enc_out, _, _ = self.modules.crisper(wavs_other, bos_tokens_other)
            enc_out_last = enc_out[-1]

            # Calculate the loss between the PCmic output and the other channels' output
            loss = self.modules.mse(enc_out_last, enc_out_ref_last)
            total_loss += loss
            num_comparisons += 1

        # Final loss as the average of all comparisons
        final_loss = total_loss / num_comparisons if num_comparisons > 0 else None

        hyps = None
        return hyps, wav_lens, final_loss*1000


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (hyps, wav_lens, loss) = predictions
        batch = batch.to(self.device)
        
        ids = batch.id
        channel_ids, _ = batch.channel_id_encoded
        
        # channel_id == 2 as PCmic
        logger.info(f"loss: {loss}")
        return loss
    
    
    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        
        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            # else:
            #     outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            #     loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()
        
        if should_step:
            self.optimizers_step()

        # self.on_fit_batch_end(self.gradient_norm, loss, should_step)
        self.on_fit_batch_end(loss, should_step)
        return loss.detach().cpu()
    
    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                out = self.compute_forward(batch, stage=stage)
                loss = self.compute_objectives(out, batch, stage=stage)
        # else:
        #     out = self.compute_forward(batch, stage=stage)
        #     loss, , loss_clf = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()


    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
        
        # if stage == sb.Stage.TEST:
        #     # Show at on_stage_end of test
        #     # a list to accomondate dictionaries
        #     # self.pooled_embedding_list = []
        #     self.y_true = [] 
        #     self.y_pred = []
            
    # def on_fit_batch_end(self, gradient_norm, loss, should_step):
    def on_fit_batch_end(self, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        
        lr = self.hparams.lr_annealing_whisper.current_lr
        if should_step:
            self.hparams.lr_annealing_whisper(self.optimizer)

        if sb.utils.distributed.if_main_process():
            stage_stats = {"loss": loss}
            
            self.hparams.train_logger.log_stats(
                stats_meta={"(batch)learning rate": lr},
                train_stats=stage_stats,
            )
            # wandb.log({"train/loss": loss, "learning rate": lr, "gradient_norm": gradient_norm})
            wandb.log({"train/loss": loss, "learning rate": lr})

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            avg_train_ = 0.0
            # avg_train_loss_clf = 0.0
            if self.profiler is not None:
                self.profiler.start()
                
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss  = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )

                t.set_postfix(train_loss=self.avg_train_loss)

                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info(
                            "The profiler finished, training is stopped."
                        )
                        self.profiler.stop()
                        quit()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if self._should_save_intra_epoch_ckpt(
                    last_ckpt_time, steps_since_ckpt
                ):
                    # Checkpointer class will handle running this on main only
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
                    steps_since_ckpt = 0

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {
            "loss": stage_loss
        }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            
            # stage_stats["VALID/CER"] = self.cer_metric.summarize("error_rate")
            # stage_stats["Valid/ErrorRate"] = self.error_metrics.summarize("average")
            
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={
                    "Valid/Loss": stage_stats["loss"],
                },
                min_keys=["Valid/Loss"],
            )
            
            # wandb.log({"valid/cer": stage_stats["VALID/CER"]})
            # wandb.log({"valid/error rate": stage_stats["Valid/ErrorRate"]}) 
            wandb.log({"valid/loss": stage_stats["loss"]})
            
        elif stage == sb.Stage.TEST:
            # scatteredness = cal_scatteredness(self.pooled_embedding_list, self.hparams.output_folder)
            
            # stage_stats["Test/ErrorRate"] = self.error_metrics.summarize("average")
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            # if if_main_process():
            #     with open(self.hparams.test_channel_file, "w") as w:
            #         self.error_metrics.write_stats(w)

            # wandb.log({"test/cer": stage_stats["TEST/CER"]})
            # wandb.log({"scatteredness": scatteredness})   
            # wandb.log({"test/error rate": stage_stats["Test/ErrorRate"]})
            wandb.log({"test/loss": stage_stats["loss"]})
             
            
    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(sb.Stage.VALID, epoch)
            self.modules.eval()
            
            avg_valid_loss = 0.0
            
            with torch.no_grad():
                
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss  = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break
                self.step = 0
                self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)            
    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        
        avg_test_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss  = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss 
            
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )
        test_dataset_representation = test_datasets[name]
        
    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_stem = sb.dataio.encoder.CategoricalEncoder()
    
    # 3. Define classifier pipeline:
    @sb.utils.data_pipeline.takes("channel")
    @sb.utils.data_pipeline.provides("channel", "channel_id_encoded")
    def label_pipeline(channel):
        yield channel
        channel_id_encoded = label_encoder.encode_sequence_torch([channel])
        yield channel_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    
    @sb.utils.data_pipeline.takes("stem")
    @sb.utils.data_pipeline.provides("stem", "stem_encoded")
    def stem_pipeline(stem):
        yield stem
        stem_encoded = label_encoder_stem.encode_sequence_torch([stem])
        yield stem_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, stem_pipeline)
    
    # 4. Define text pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides(
        "text", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(text):
        if hasattr(hparams, "normalized_transcripts"):
            text = tokenizer.normalize(text)
        yield text
        tokens_list = tokenizer.encode(text, add_special_tokens=False)
        yield tokens_list
        
        # special tokens build here
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
        
    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path="label_encoder_3.txt",
        from_didatasets=[train_data],
        output_key="channel",
    )
    
    lab_enc_file_stem = os.path.join(hparams["save_folder"], "label_encoder_stem.txt")
    
    label_encoder_stem.load_or_create(
        path=lab_enc_file_stem,
        from_didatasets=[train_data, valid_data, test_dataset_representation],
        output_key="stem",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "channel_id_encoded", "stem_encoded"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    # wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"crisper_enc_epo_{hparams['number_of_epochs']}_{hparams['seed']}_0806",

        # track hyperparameters and run metadata
        config = {
            "channel": hparams["channel"],
            "seed": hparams["seed"],
            "output_folder": hparams["output_folder"],
            "whisper_folder": hparams["whisper_folder"],
            "crisper_folder": hparams["crisper_folder"],
            "normalized_transcripts": hparams["normalized_transcripts"],
            "data_folder": hparams["data_folder"],
            "train_csv": hparams["train_csv"],
            "valid_csv": hparams["valid_csv"],
            "test_csv": hparams["test_csv"],
            "freeze_encoder": hparams["freeze_encoder"],
            "freeze_blocks": hparams["freeze_blocks"],
            "freeze_decoder": hparams["freeze_decoder"],
            "number_of_epochs": hparams["number_of_epochs"],
            "warmup_steps": hparams["warmup_steps"],
            "sorting": hparams["sorting"],
            "batch_size": hparams["batch_size"],
            "test_batch_size": hparams["test_batch_size"],
            "grad_accumulation_factor": hparams["grad_accumulation_factor"],
        }
    )   

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["crisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    crisper = Crisper(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(crisper.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    crisper.tokenizer = tokenizer
    
    logger.info("========================= Start Training =========================")
    
    # Training
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        crisper.fit(
            crisper.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"],
        )

    # Testing
    # os.makedirs(hparams["output_cer_folder"], exist_ok=True)
    
    logger.info("========================= Start Testing =========================")
    for k in test_datasets.keys():  # keys are test_clean, test_other etc

        crisper.hparams.test_channel_file = os.path.join(
            hparams["output_channel_folder"], f"channel_preds_{k}.txt"
        )
        
        crisper.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="Valid/Loss",
        )