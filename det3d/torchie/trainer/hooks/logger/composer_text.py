import datetime
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist
from det3d import torchie

from .base import LoggerHook


class ComposerTextLoggerHook(LoggerHook):
    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(ComposerTextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, trainer):
        super(ComposerTextLoggerHook, self).before_run(trainer)
        self.start_iter = trainer.iter
        self.json_log_path = osp.join(
            trainer.work_dir, "{}.log.json".format(trainer.timestamp)
        )

    def _get_max_memory(self, trainer):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor(
            [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
        )
        if trainer.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _convert_to_precision4(self, val):
        if isinstance(val, float):
            val = "{:.4f}".format(val)
        elif isinstance(val, list):
            val = [self._convert_to_precision4(v) for v in val]

        return val

    def _log_info(self, log_dict, trainer):
        if trainer.mode == "train":
            log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
                log_dict["epoch"],
                trainer._max_epochs,
                log_dict["iter"],
                len(trainer.data_loader),
                log_dict["lr"],
            )
            if "time" in log_dict.keys():
                self.time_sec_tot += log_dict["time"] * self.interval
                time_sec_avg = self.time_sec_tot / (trainer.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (trainer.max_iters - trainer.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += "eta: {}, ".format(eta_str)
                log_str += "time: {:.3f}, data_time: {:.3f}, transfer_time: {:.3f}, forward_time: {:.3f}, loss_parse_time: {:.3f} ".format(
                    log_dict["time"],
                    log_dict["data_time"],
                    log_dict["transfer_time"] - log_dict["data_time"],
                    log_dict["forward_time"] - log_dict["transfer_time"],
                    log_dict["loss_parse_time"] - log_dict["forward_time"],
                )
                log_str += "memory: {}, ".format(log_dict["memory"])
        else:
            log_str = "Epoch({}) [{}][{}]\t".format(
                log_dict["mode"], log_dict["epoch"] - 1, log_dict["iter"]
            )

        trainer.logger.info(log_str)

        for prefix, task in zip(['gen', 'dsc'], ['Generator', 'Discriminator']):
            log_str = ""
            log_items = [f"Module : {task}"]

            for loss_name, loss_value in log_dict.items():
                if not loss_name.startswith(prefix):
                    continue
                loss_name = loss_name[4:]
                assert len(loss_value) == 1

                if len(loss_value) == 1:
                    loss_value = self._convert_to_precision4(loss_value[0])
                else:
                    loss_value = self._convert_to_precision4(loss_value)
                log_items.append(
                    f"{loss_name}: {loss_value}"
                )
            
            log_str += ", ".join(log_items)
            trainer.logger.info(log_str)

    def _dump_log(self, log_dict, trainer):
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)

        if trainer.rank == 0:
            with open(self.json_log_path, "a+") as f:
                torchie.dump(json_log, f, file_format="json")
                f.write("\n")

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, trainer):
        log_dict = OrderedDict()
        # Training mode if the output contains the key time
        mode = "train" if "time" in trainer.log_buffer.output else "val"
        log_dict["mode"] = mode
        log_dict["epoch"] = trainer.epoch + 1
        log_dict["iter"] = trainer.inner_iter + 1
        # Only record lr of the first param group
        log_dict["lr"] = trainer.current_lr()[0]
        if mode == "train":
            log_dict["time"] = trainer.log_buffer.output["time"]
            log_dict["data_time"] = trainer.log_buffer.output["data_time"]
            # statistic memory
            if torch.cuda.is_available():
                log_dict["memory"] = self._get_max_memory(trainer)
        for name, val in trainer.log_buffer.output.items():
            if name in ["time", "data_time"]:
                continue
            log_dict[name] = val

        self._log_info(log_dict, trainer)
        self._dump_log(log_dict, trainer)
