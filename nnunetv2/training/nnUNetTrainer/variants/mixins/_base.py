"""
Base classes for the composable mixin trainer architecture.

``TrainerMixin`` defines no-op hook methods that serve as chain terminators.
``ComposableTrainerMixin`` overrides real nnUNetTrainer methods and dispatches
to the hook chain so that feature mixins can inject behaviour without
duplicating the training loop.

MRO example::

    DiseaseCondMixin -> TopologyLossMixin -> ComposableTrainerMixin -> DA5/Base

Each mixin implements ``mixin_*`` hooks that chain via ``super()``.
"""
from __future__ import annotations

import warnings
from os.path import join
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from nnunetv2.utilities.helpers import dummy_context


def _blosc2_init_worker():
    """Module-level pool initializer so it can be pickled by spawn context."""
    import blosc2
    blosc2.set_nthreads(1)


# =========================================================================
# TrainerMixin — no-op hook terminators
# =========================================================================

class TrainerMixin:
    """Base class providing no-op hook methods.

    Every hook returns a neutral value so that mixins further up the MRO
    can chain via ``super()`` without worrying about whether another mixin
    is present.
    """

    def mixin_init(self):
        """Called at the end of ``__init__`` for setting mixin-specific attrs."""
        pass

    def mixin_initialize(self):
        """Called at the end of ``initialize()`` (post-setup)."""
        pass

    def mixin_prepare_forward(self, batch: dict) -> dict:
        """Return extra kwargs to pass to ``self.network(data, **extra)``."""
        return {}

    def mixin_modify_target(self, target, batch: dict):
        """Modify target labels before loss computation (training only).

        Must return the (possibly modified) target.  Default: passthrough.

        NOTE: this hook fires only in ``train_step``, not ``validation_step``.
        Validation is intentionally evaluated against the *unmodified* targets
        so val Dice tracks the true labelling regardless of any train-time
        target merging (e.g. disease-adaptive class collapsing).
        """
        return target

    def mixin_extra_loss(self, output, target, batch: dict, **forward_kwargs) -> float:
        """Return an additional scalar loss term (0.0 = no extra loss)."""
        return 0.0

    def mixin_param_groups(self) -> List[dict]:
        """Return extra optimizer param groups with custom LR."""
        return []

    def mixin_fix_lr_after_scheduler(self):
        """Re-apply LR multipliers after the scheduler steps all groups."""
        pass

    def before_validation_case(self, case_id: str):
        """Per-case setup before sliding window prediction."""
        pass

    def after_validation_case(self, case_id: str):
        """Per-case cleanup after sliding window prediction."""
        pass

    def mixin_on_train_start(self):
        """Called at the end of ``on_train_start()``."""
        pass

    def mixin_on_train_epoch_start(self):
        """Called at the end of ``on_train_epoch_start()``."""
        pass

    def mixin_on_train_epoch_end(self, train_outputs):
        """Called at the end of ``on_train_epoch_end()``."""
        pass

    def mixin_checkpoint_extras(self) -> dict:
        """Return extra state to persist in checkpoint (not used yet)."""
        return {}


# =========================================================================
# ComposableTrainerMixin — dispatches trainer methods to hook chain
# =========================================================================

class ComposableTrainerMixin(TrainerMixin):
    """Overrides real nnUNetTrainer methods to dispatch to mixin hooks.

    Must appear in the MRO *after* all feature mixins and *before* the
    concrete base trainer (e.g. nnUNetTrainerDA5).
    """

    # Gradient clipping max norm; mirrors nnUNetTrainer's hardcoded 12 but is
    # exposed as a class attribute so mixins/subclasses can override (e.g. an
    # aggressive topology run might want a tighter clip).
    gradient_clip_max_norm: float = 12.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixin_init()

    def initialize(self):
        super().initialize()
        self.mixin_initialize()

    # ------------------------------------------------------------------
    # Helper: get unwrapped network module
    # ------------------------------------------------------------------
    def _get_unwrapped_network(self) -> nn.Module:
        mod = self.network
        if self.is_ddp:
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        forward_kwargs = self.mixin_prepare_forward(batch)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            result = self.network(data, **forward_kwargs)
            # wrapper networks may return (seg_output, aux_logits)
            if isinstance(result, tuple):
                output, aux_logits = result
            else:
                output, aux_logits = result, None

            # let mixins modify targets before loss (e.g. disease-adaptive merging)
            target = self.mixin_modify_target(target, batch)

            l = self.loss(output, target)

            # let mixins add extra loss terms
            extra = self.mixin_extra_loss(
                output, target, batch,
                aux_logits=aux_logits,
                **forward_kwargs,
            )
            if extra != 0.0:
                l = l + extra

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_max_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_max_norm)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        forward_kwargs = self.mixin_prepare_forward(batch)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, **forward_kwargs)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = target != self.label_manager.ignore_label
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = (1 - target[:, -1:]).bool()
                target = target[:, :-1].bool()
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    # ------------------------------------------------------------------
    # Optimizer: merge mixin param groups into the base optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        extra_groups = self.mixin_param_groups()
        if not extra_groups:
            return super().configure_optimizers()

        # collect param ids from extra groups to exclude from main group
        extra_param_ids = set()
        for group in extra_groups:
            for p in group["params"]:
                extra_param_ids.add(id(p))

        main_params = [p for p in self.network.parameters() if id(p) not in extra_param_ids]

        all_groups = [{"params": main_params}] + extra_groups

        optimizer = torch.optim.SGD(
            all_groups,
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()
        self.mixin_on_train_start()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.mixin_fix_lr_after_scheduler()
        self.mixin_on_train_epoch_start()

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self.mixin_on_train_epoch_end(train_outputs)

    # ------------------------------------------------------------------
    # perform_actual_validation: with before/after hooks per case
    # ------------------------------------------------------------------
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work."
            )

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
            resample_and_save
        from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
        from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.utilities.helpers import empty_cache
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
        import multiprocessing
        import torch.distributed as dist
        from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p
        default_num_processes = 8

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes, initializer=_blosc2_init_worker) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                # === mixin hook: before validation case ===
                self.before_validation_case(k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # === mixin hook: after validation case ===
                self.after_validation_case(k)

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)
                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue
                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager, properties, self.dataset_json,
                                 default_num_processes, dataset_class),
                            )
                        ))

                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    # ------------------------------------------------------------------
    # Checkpoint loading (strict=False for partial loading)
    # ------------------------------------------------------------------
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )

        mod = self._get_unwrapped_network()
        missing, unexpected = mod.load_state_dict(new_state_dict, strict=False)
        if missing:
            self.print_to_log_file(
                f"load_checkpoint: missing keys (will use init weights): {missing}"
            )
        if unexpected:
            self.print_to_log_file(
                f"load_checkpoint: unexpected keys (ignored): {unexpected}"
            )

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
