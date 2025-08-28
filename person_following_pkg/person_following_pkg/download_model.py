from collections import OrderedDict
from pathlib import Path
import time
import urllib.request
import sys
import warnings

import torch

__model_types = [
    "resnet50",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
]

__trained_urls = {
    # [Ton dictionnaire __trained_urls est intact, je ne le recopie pas ici pour alléger]
}


def show_downloadeable_models():
    print("\nAvailable .pt ReID models for automatic download")
    print(list(__trained_urls.keys()))


def get_model_url(model):
    if isinstance(model, (str, Path)):
        key = model
    elif hasattr(model, 'name'):
        key = model.name
    else:
        key = None

    if key is not None and key in __trained_urls:
        return __trained_urls[key]
    else:
        return None


def is_model_in_model_types(model):
    return model.name in __model_types


def get_model_name(model):
    for x in __model_types:
        model_name = Path(model).name if isinstance(model, (str, Path)) else model.name
        if x in model_name:
            return x
    return None


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (
                percent,
                progress_size / (1024 * 1024),
                speed,
                duration
            )
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrained weights to model.

    Features:
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples:
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = torch.load(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f'The pretrained weights "{weight_path}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)"
        )
    else:
        print(f'Successfully loaded pretrained weights from "{weight_path}"')
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".format(discarded_layers)
            )
