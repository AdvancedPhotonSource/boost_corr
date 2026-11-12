import numpy as np
import logging
import torch
from .xpcs_dataset import XpcsDataset
from timepix_dataset.dataset import TimepixRawDataset
from ...devices import is_gpu_device
import boost_corr.xpcs_aps_8idi.exceptions as exc


logger = logging.getLogger(__name__)


class TimepixAnalysisDataset(XpcsDataset):
    def __init__(
        self,
        *args,
        dtype=np.uint8,
        total_frames=None,
        bin_time_s=1e-6,
        run_config_path=None,
        device="cpu",
        **kwargs,
    ):
        super(TimepixAnalysisDataset, self).__init__(
            *args, dtype=dtype, device=device, **kwargs
        )
        # base class's default (1 second) would otherwise silently win: this
        # subclass declares bin_time_s as its own explicit kwarg (needed
        # below, for read_data), so it never reaches **kwargs and never
        # reaches super().__init__() above.
        self.bin_time_s = bin_time_s
        self.dataset_type = "Timepix4Dataset"
        self.is_sparse = True
        self.dtype = np.uint8
        self.raw_dataset = TimepixRawDataset(self.fname, run_config_path, device=device)
        self.update_det_size(self.raw_dataset.det_size)

        self.ifc_list, self.mem_addr_list = self.read_data(
            total_frames, bin_time_s=bin_time_s
        )
        self.ifc_list = self.to_device()

    def to_device(self, max_fsize_gb_in_gpu=1.0):
        ifc_device = []
        for ifc in self.ifc_list:
            module_ifc = self.to_device_single_module(ifc, max_fsize_gb_in_gpu)
            ifc_device.append(module_ifc)
        return ifc_device

    def to_device_single_module(self, ifc, max_fsize_gb_in_gpu=1.0):
        total_size = sum([arr.nbytes for arr in ifc]) / 1024**3
        if total_size <= max_fsize_gb_in_gpu and is_gpu_device(self.device):
            logger.info("Preloading timepix rawdata to GPU memory")
            device = self.device
        else:
            device = "cpu"
            logger.info(f"Total data size {total_size:.2f} GB, offloading to CPU RAM")
        index = torch.tensor(ifc[0], device=device)
        frame = torch.tensor(ifc[1], device=device)
        count = torch.tensor(ifc[2], device=device)
        return (index, frame, count)

    def read_data(self, total_frames=None, bin_time_s=1e-6):
        ifc_list = self.raw_dataset.apply_time_binning(bin_time_s)

        # the last frame is dropped to avoid partial frames
        num_frames = max([frame[-1] for _, frame, _ in ifc_list])
        # update frame_num and batch_num
        self.update_batch_info(num_frames)

        all_index = np.arange(0, self.frame_num_raw + 2)
        all_index[-1] = self.frame_num_raw

        mem_addr_list = []
        for ifc in ifc_list:
            # build an index map for all indexes
            beg = np.searchsorted(ifc[1], all_index[:-1])
            end = np.searchsorted(ifc[1], all_index[1:])
            mem_addr = [slice(a, b) for a, b in zip(beg, end)]
            mem_addr_list.append(mem_addr)
        return ifc_list, mem_addr_list

    def __getbatch_single_module__(self, idx, module_index=0):
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        ifc = self.ifc_list[module_index]
        device = ifc[0].device

        pixel_num = self.raw_dataset.mod_size[0] * self.raw_dataset.mod_size[1]
        x = torch.zeros(size, pixel_num, dtype=torch.bfloat16, device=device)
        if self.stride == 1:
            # the data is continuous in RAM; convert by batch
            sla, slb = (
                self.mem_addr_list[module_index][beg],
                self.mem_addr_list[module_index][end],
            )
            a, b = sla.start, slb.start
            x[ifc[1][a:b].long() - beg, ifc[0][a:b].long()] = ifc[2][a:b].to(
                torch.bfloat16
            )
        else:
            # the data is discrete in RAM; convert frame by frame
            for n, idx in enumerate(np.arange(beg, end, self.stride)):
                sl = self.mem_addr_list[idx]
                x[n, ifc[0][sl].long()] = ifc[2][sl].to(torch.bfloat16)

        x = x.to(self.device)
        return x

    def __getbatch__(self, idx):
        # deal with with single modules
        if self.raw_dataset.num_chips == 1:
            canvas = self.__getbatch_single_module__(idx, module_index=0)
        # deal with with multiple modules
        else:
            canvas = torch.zeros(
                self.batch_size,
                *self.raw_dataset.det_size,
                dtype=torch.uint8,
                device=self.device,
            )
            for module_index in range(self.raw_dataset.num_chips):
                _data = self.__getbatch_single_module__(idx, module_index=module_index)
                actual_batch_size = _data.shape[0]

                _data = _data.to(self.device)
                _data = _data.view(actual_batch_size, *self.raw_dataset.mod_size)
                layout = self.raw_dataset.layout[module_index]
                canvas[:actual_batch_size, :, layout] = _data

            # print(canvas.shape) # canvas is 3d
            # disabled for now: signal-spill correction is superseded by the
            # event correlator's CCF direction mechanism when --signal-spill
            # is set (see EventCorrelator). Kept, not deleted, for
            # reference/future reactivation.
            # canvas = self.update_signal_spill(canvas)
            canvas = canvas[0:actual_batch_size].view(actual_batch_size, self.pixel_num)

        if self.mask_crop is not None:
            canvas = canvas[:, self.mask_crop]

        return canvas

    def get_events(self):
        """Return (pixel, time_bin, count, bin_width_ns) for the event
        correlator. Single-chip datasets return their pixel index as-is;
        3-chip datasets remap each module's own pixel index into the flat
        full-canvas index space via raw_dataset.layout -- the same
        column-shift __getbatch__ uses to stitch the dense frame.

        ACF pairs (same pixel) and CCF neighbour pairs (exactly +/-1 pixel
        offset, see _event_core.neighbour_pairs) can never span two chips
        for this geometry: the real inter-chip gaps are >=4 columns wide
        and never carry events, so the cross-module clock-sync risk noted
        in docs/superpowers/specs/2026-09-17-event-correlator-integration-
        design.md does not apply here.
        """
        num_chips = self.raw_dataset.num_chips
        if num_chips not in (1, 3):
            raise exc.DatasetError(
                "get_events() only supports single-chip or 3-chip Timepix4 "
                f"datasets (this dataset has {num_chips} chips); use "
                "--signal-spill none to run multitau/twotime instead."
            )
        if num_chips == 1:
            index, frame, count = self.ifc_list[0]
            pixel = index.detach().cpu().numpy().astype(np.int64)
            time_bin = frame.detach().cpu().numpy().astype(np.int64)
            count = count.detach().cpu().numpy().astype(np.int64)
        else:
            mod_w = self.raw_dataset.mod_size[1]
            det_w = self.raw_dataset.det_size[1]
            pixel_parts, time_parts, count_parts = [], [], []
            for module_index, (index, frame, cnt) in enumerate(self.ifc_list):
                idx = index.detach().cpu().numpy().astype(np.int64)
                row, col = np.divmod(idx, mod_w)
                col_offset = self.raw_dataset.layout[module_index].start
                pixel_parts.append(row * det_w + col + col_offset)
                time_parts.append(frame.detach().cpu().numpy().astype(np.int64))
                count_parts.append(cnt.detach().cpu().numpy().astype(np.int64))
            pixel = np.concatenate(pixel_parts)
            time_bin = np.concatenate(time_parts)
            count = np.concatenate(count_parts)
        bin_width_ns = self.bin_time_s * 1e9
        return pixel, time_bin, count, bin_width_ns
