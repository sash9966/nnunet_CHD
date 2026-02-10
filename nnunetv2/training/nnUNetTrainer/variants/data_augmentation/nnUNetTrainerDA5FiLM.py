"""
DA5 trainer variant with FiLM disease-vector conditioning.

Loads ``disease_map.json`` from the preprocessed dataset folder and injects
a per-sample disease vector into the network via Feature-wise Linear
Modulation (FiLM) at every forward pass.  When the JSON is absent, training
falls back to exact baseline behaviour.

During training the disease map is copied to the model output folder so that
inference scripts can find it automatically.
"""
from __future__ import annotations

import json
import shutil
import warnings
from os.path import isfile, join
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerDA5FiLM(nnUNetTrainerDA5):
    """nnUNetTrainerDA5 + FiLM disease-vector conditioning.

    Configuration knobs (override in subclass ``__init__`` if needed):

    ============  ========  ===================================================
    Attribute     Default   Meaning
    ============  ========  ===================================================
    disease_K     8         Length of the binary disease flag vector
    disease_H     64        Hidden dimension of the disease MLP
    disease_E     32        Output embedding dimension of the disease MLP
    ============  ========  ===================================================
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # disease conditioning hyper-parameters
        self.disease_K: int = 8
        self.disease_H: int = 64
        self.disease_E: int = 32
        # populated in initialize()
        self.disease_map: Optional[Dict[str, List[int]]] = None

    # ------------------------------------------------------------------
    # Disease map loading
    # ------------------------------------------------------------------
    def _load_disease_map(self) -> Optional[Dict[str, List[int]]]:
        """Load ``disease_map.json`` from the preprocessed dataset base folder.

        Returns ``None`` if the file does not exist (baseline mode).
        """
        path = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if not isfile(path):
            self.print_to_log_file(
                f"WARNING: {path} not found. Disease conditioning DISABLED (baseline mode)."
            )
            return None
        with open(path) as f:
            disease_map = json.load(f)
        self.print_to_log_file(
            f"Loaded disease_map.json with {len(disease_map)} entries from {path}"
        )
        for case_id, vec in disease_map.items():
            assert isinstance(vec, list) and len(vec) == self.disease_K, (
                f"disease_map entry for '{case_id}' has length {len(vec)}, expected {self.disease_K}"
            )
        return disease_map

    # ------------------------------------------------------------------
    # Network building  (override the @staticmethod as an instance method)
    # ------------------------------------------------------------------
    def build_network_architecture(  # type: ignore[override]
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the base network then wrap it with FiLM conditioning."""
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        wrapped = FiLMConditionedResEncUNet(
            base_network,
            disease_K=self.disease_K,
            disease_H=self.disease_H,
            disease_E=self.disease_E,
        )

        # He-init the disease MLP (FiLM heads are already zero-init)
        wrapped.disease_mlp.apply(FiLMConditionedResEncUNet.initialize)

        return wrapped

    # ------------------------------------------------------------------
    # Initialization (load disease map after parent sets up everything)
    # ------------------------------------------------------------------
    def initialize(self):
        super().initialize()
        self.disease_map = self._load_disease_map()

    # ------------------------------------------------------------------
    # on_train_start: copy disease_map.json to model output folder
    # ------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()
        # copy disease_map.json so inference scripts can find it in the model folder
        src = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if isfile(src):
            dst = join(self.output_folder_base, "disease_map.json")
            shutil.copy2(src, dst)
            self.print_to_log_file(f"Copied disease_map.json to {dst}")

    # ------------------------------------------------------------------
    # Helper: set disease_vec on the network for a single case
    # ------------------------------------------------------------------
    def _set_disease_vec_for_case(self, case_id: str) -> None:
        """Set disease_vec on the network for inference of a single case."""
        mod = self._get_unwrapped_network()
        if self.disease_map is not None and case_id in self.disease_map:
            vec = torch.tensor([self.disease_map[case_id]], dtype=torch.float32, device=self.device)
            mod.set_disease_vec(vec)
        else:
            mod.clear_disease_vec()

    # ------------------------------------------------------------------
    # perform_actual_validation: set disease_vec per case during inference
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

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
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

                # === disease conditioning: set vec for this case ===
                self._set_disease_vec_for_case(k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # === clear disease vec ===
                self._get_unwrapped_network().clear_disease_vec()

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
    # Helper: build disease_vec tensor for a batch
    # ------------------------------------------------------------------
    def _build_disease_vec(self, batch_keys: list) -> Optional[torch.Tensor]:
        """Return (B, K) float tensor or ``None`` if conditioning is disabled."""
        if self.disease_map is None:
            return None
        vecs = []
        for case_id in batch_keys:
            if case_id not in self.disease_map:
                raise KeyError(
                    f"Case '{case_id}' not found in disease_map.json. "
                    "All training/validation cases must be present."
                )
            vecs.append(self.disease_map[case_id])
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)

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

        disease_vec = self._build_disease_vec(batch["keys"])

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, disease_vec=disease_vec)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Validation step  (DA5 overrides this, so we override it here too)
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        disease_vec = self._build_disease_vec(batch["keys"])

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, disease_vec=disease_vec)
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
    # Checkpoint loading  (handle missing FiLM keys gracefully)
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


class nnUNetTrainerDA5FiLM_100epochs(nnUNetTrainerDA5FiLM):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
