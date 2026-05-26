import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
import numpy as np

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
        # Optional per-class plot configuration (set by trainers)
        self.label_names = None       # list of str, e.g. ["LV", "RV", ...]
        self.topo_class_ids = None    # list of int — highlighted with thicker lines
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        # auto-create new logging keys (for mixin-injected diagnostics like FiLM magnitudes)
        if key not in self.my_fantastic_logging:
            self.my_fantastic_logging[key] = list()

        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    # core keys used for epoch inference (mixin-added keys excluded)
    _CORE_KEYS = {'mean_fg_dice', 'ema_fg_dice', 'dice_per_class_or_region',
                  'train_losses', 'val_losses', 'lrs',
                  'epoch_start_timestamps', 'epoch_end_timestamps'}

    def plot_progress_png(self, output_folder):
        # we infer the epoch from core logging keys only (extra keys may be shorter)
        core_lengths = [len(v) for k, v in self.my_fantastic_logging.items() if k in self._CORE_KEYS]
        epoch = min(core_lengths) - 1 if core_lengths else 0  # lists of epoch 0 have len 1

        has_film = ('film_magnitudes' in self.my_fantastic_logging
                    and len(self.my_fantastic_logging['film_magnitudes']) > 0)
        has_aux = ('aux_diagnosis' in self.my_fantastic_logging
                   and len(self.my_fantastic_logging['aux_diagnosis']) > 0)
        n_plots = 4 + int(has_film) + int(has_aux)

        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(n_plots, 1, figsize=(30, 18 * n_plots))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # per-class validation Dice
        ax = ax_all[3]
        dice_data = self.my_fantastic_logging['dice_per_class_or_region'][:epoch + 1]
        if len(dice_data) > 0 and dice_data[0] is not None:
            num_classes = len(dice_data[0])
            cmap = plt.cm.get_cmap('tab10', max(num_classes, 10))
            for c in range(num_classes):
                class_dice = []
                for d in dice_data:
                    if d is not None and c < len(d):
                        v = d[c]
                        class_dice.append(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else float('nan'))
                    else:
                        class_dice.append(float('nan'))
                # Label: use named labels if available, else generic
                if self.label_names is not None and c < len(self.label_names):
                    class_name = self.label_names[c]
                else:
                    class_name = f"Class {c + 1}"
                # Highlight topology-relevant classes with thicker solid lines
                if self.topo_class_ids is not None and c < num_classes:
                    # topo_class_ids are label indices; foreground class c maps to label c+1
                    # (background removed from dice_per_class_or_region)
                    fg_label = c + 1  # offset because background is excluded
                    is_topo = fg_label in self.topo_class_ids
                else:
                    is_topo = False
                lw = 4 if is_topo else 1.5
                ls = '-' if is_topo else '--'
                ax.plot(x_values[:len(class_dice)], class_dice,
                        color=cmap(c % 10), ls=ls, label=class_name, linewidth=lw)
            ax.set_ylim([0, 1])
            ax.legend(loc='lower right', fontsize=14, ncol=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("pseudo Dice")
        ax.set_title("Per-class validation pseudo Dice")

        # FiLM modulation magnitude diagnostic (only for FiLM trainers)
        if has_film:
            ax = ax_all[4]
            film_data = self.my_fantastic_logging['film_magnitudes']
            film_epoch = min(len(film_data), epoch + 1)
            film_x = list(range(film_epoch))

            if film_epoch > 0:
                # collect all layer names from the first entry
                layer_names = sorted(film_data[0].keys())
                cmap = plt.cm.get_cmap('tab10', max(len(layer_names) * 2, 10))
                for li, layer_name in enumerate(layer_names):
                    gammas = []
                    betas = []
                    for ep in range(film_epoch):
                        entry = film_data[ep] if ep < len(film_data) else {}
                        layer_data = entry.get(layer_name, {})
                        gammas.append(layer_data.get("gamma", float('nan')))
                        betas.append(layer_data.get("beta", float('nan')))
                    color = cmap(li % 10)
                    ax.plot(film_x, gammas, color=color, ls='-',
                            label=f"|gamma| {layer_name}", linewidth=3)
                    ax.plot(film_x, betas, color=color, ls='--',
                            label=f"|beta| {layer_name}", linewidth=2)

            ax.set_xlabel("epoch")
            ax.set_ylabel("mean |value|")
            ax.set_title("FiLM modulation magnitudes (should grow > 0.01 if conditioning is active)")
            ax.legend(loc='upper left', fontsize=12, ncol=2)
            ax.set_ylim(bottom=0)

        # Auxiliary diagnosis head diagnostic (only for AuxDiag trainers)
        if has_aux:
            ax = ax_all[4 + int(has_film)]
            aux_data = self.my_fantastic_logging['aux_diagnosis']
            aux_epoch = min(len(aux_data), epoch + 1)
            aux_x = list(range(aux_epoch))

            # Left axis: BCE loss
            bce_vals = [aux_data[e]['bce'] for e in range(aux_epoch)]
            ax.plot(aux_x, bce_vals, color='black', ls='-', linewidth=3,
                    label='aux BCE loss')
            ax.set_xlabel("epoch")
            ax.set_ylabel("aux BCE loss")
            ax.set_ylim(bottom=0)

            # Right axis: per-class mean predicted probability
            ax_r = ax.twinx()
            disease_names = ['HLHS', 'ASD', 'VSD', 'AVSD', 'DORV', 'PuA', 'ToF', 'TGA']
            cmap = plt.cm.get_cmap('tab10', 8)
            for c in range(8):
                probs = []
                for e in range(aux_epoch):
                    p = aux_data[e].get('probs', [])
                    probs.append(p[c] if c < len(p) else float('nan'))
                label = disease_names[c] if c < len(disease_names) else f"class {c}"
                ax_r.plot(aux_x, probs, color=cmap(c), ls='--', linewidth=2, label=label)
            ax_r.axhline(0.5, color='gray', ls=':', linewidth=1)
            ax_r.set_ylabel("mean predicted probability")
            ax_r.set_ylim([0, 1])

            ax.set_title(
                "Aux diagnosis head — BCE loss (black, left) + mean pred prob per disease (dashed, right)\n"
                "BCE should decrease; probs should differ from 0.5 if the head is learning"
            )
            ax.legend(loc='upper right')
            ax_r.legend(loc='upper left', fontsize=11, ncol=2)

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
