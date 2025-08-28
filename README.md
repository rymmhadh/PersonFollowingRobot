# Person-Following-Robot

This project implements a real-time **person following system** using:
- **YOLOv8** for person detection
- **StrongSORT** for multi-object tracking
- **FaceNet** (InceptionResnetV1) for facial recognition
- **TorchReID (OSNet)** for body re-identification
- **MongoDB** for storing person profiles

The system allows you to:
1. Detect and track people in real time from a webcam.
2. Select a person to record for 20 seconds.
3. Automatically extract and average face & body embeddings.
4. Save the profile to a database.
5. Re-identify and track the target later using embeddings.

---

## Features
- Real-time YOLOv8 person detection
- Multi-object tracking with StrongSORT
- Face embedding extraction via FaceNet
- Front and rear body embedding extraction via TorchReID
- Profile storage in MongoDB
- Click-to-select recording mechanism
- Embedding-based re-identification

---

## Requirements
- **Python 3.10+**
- **CUDA-compatible GPU** 
- **MongoDB** running locally or remotely

---

## Installation


2. Create and activate a virtual environment
python -m venv .venv-py39
.\.venv-py39\Scripts\Activate.ps1


# Linux/Mac
python3 -m venv .venv-py39
source .venv-py39/bin/activate


3. Install PyTorch with CUDA 12.1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install yolov5
pip install cython
pip install opencv-python==4.6.0.66
pip install facenet-pytorch==2.6.0
pip install ultralytics==8.3.161
pip install torchreid==1.4.0
pip install strongsort
pip install gdown==4.5.3
pip install pillow
pip install numpy
pip install pymongo

----------Activate the virtual environment-----
Modification in .venv-py39/Lib/site-packages/strongsort-2.0.1.dist-info/METADATA

Requires-Dist: torch>=1.13.0
Requires-Dist: torchvision>=0.17.0

🛠 Modification in : reid_multibackend.py
To ensure compatibility with our CUDA and PyTorch environment, the file needs to be updated:

access:
.venv-py39/Lib/site-packages/ultralytics/trackers/reid_multibackend.py


from collections import OrderedDict, namedtuple
from os.path import exists as file_exists
from pathlib import Path
from pathlib import Path


import gdown
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from yolov5.utils.general import check_requirements, check_version

from strongsort.deep.models import build_model
from strongsort.deep.reid_model_factory import (
    get_model_name,
    get_model_url,
    load_pretrained_weights,
    show_downloadeable_models,
)


# kadirnar: I added export_formats to the function
def export_formats():
    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights="osnet_x0_25_msmt17.pt", device=torch.device("cpu"), fp16=False):
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.coreml,
            self.saved_model,
            self.pb,
            self.tflite,
            self.edgetpu,
            self.tfjs,
        ) = self.model_type(
            w
        )  # get backend
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine  # FP16

        # Build transform functions
        self.device = device
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()
        model_name = get_model_name(w)

        if Path(w).suffix == ".pt":
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                print(f"No URL associated to the chosen StrongSORT weights ({w}). Choose between:")
                show_downloadeable_models()
                exit()

        # Build model
        

        w_path = Path(w) if isinstance(w, str) else w
        self.model = build_model(model_name, num_classes=1, pretrained=not (w_path and w_path.is_file()), use_gpu=device)

        if self.pt:  # PyTorch
            # populate model arch with weights
            if w and w_path.is_file() and w_path.suffix == ".pt":
                load_pretrained_weights(self.model, w_path)

            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.jit:
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            cuda = torch.cuda.is_available() and device.type != "cpu"
            # check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False  # default updated below
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            batch_size = self.bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif self.xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements(("openvino",))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCWH"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            self.executable_network = ie.compile_model(
                network, device_name="CPU"
            )  # device_name="MYRIAD" for Intel NCS2
            self.output_layer = next(iter(self.executable_network.outputs))

        elif self.tflite:
            LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            self.interpreter = tf.lite.Interpreter(model_path=w)
            self.interpreter.allocate_tensors()
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Test model on random input data.
            input_data = np.array(np.random.random_sample((1, 256, 128, 3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        else:
            print("This model framework is not supported yet!")
            exit()

    @staticmethod
    def model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = list(export_formats().Suffix) + [".xml"]  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, _, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

    def _preprocess(self, im_batch):

        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.to(self.device)

        return images

    def forward(self, im_batch):

        # preprocess batch
        im_batch = self._preprocess(im_batch)

        # batch to half
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()

        # batch processing
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:  # TorchScript
            features = self.model(im_batch)
        elif self.onnx:  # ONNX Runtime
            im_batch = im_batch.cpu().numpy()  # torch to numpy
            features = self.session.run(
                [self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch}
            )[0]
        elif self.engine:  # TensorRT
            if True and im_batch.shape != self.bindings["images"].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ("images", "output"))
                self.context.set_binding_shape(i_in, im_batch.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im_batch.shape)
                self.bindings["output"].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings["images"].shape
            assert (
                im_batch.shape == s
            ), f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings["output"].data
        elif self.xml:  # OpenVINO
            im_batch = im_batch.cpu().numpy()  # FP32
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            print("Framework not supported at the moment, we are working on it...")
            exit()

        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != "cpu":
            im = [np.empty(*imgsz).astype(np.uint8)]  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup











🛠 Modification in : reid_model_factory.py
Access virtual environment:
.venv-py39/Lib/site-packages/torchreid/models/reid_model_factory.py

from collections import OrderedDict
from pathlib import Path

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
    # market1501 models ########################################################
    "resnet50_market1501.pt": "https://drive.google.com/uc?id=1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV",
    "resnet50_dukemtmcreid.pt": "https://drive.google.com/uc?id=17ymnLglnc64NRvGOitY3BqMRS9UWd1wg",
    "resnet50_msmt17.pt": "https://drive.google.com/uc?id=1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj",
    "resnet50_fc512_market1501.pt": "https://drive.google.com/uc?id=1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt",
    "resnet50_fc512_dukemtmcreid.pt": "https://drive.google.com/uc?id=13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx",
    "resnet50_fc512_msmt17.pt": "https://drive.google.com/uc?id=1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud",
    "mlfn_market1501.pt": "https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS",
    "mlfn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum",
    "mlfn_msmt17.pt": "https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-",
    "hacnn_market1501.pt": "https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF",
    "hacnn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH",
    "hacnn_msmt17.pt": "https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ",
    "mobilenetv2_x1_0_market1501.pt": "https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp",
    "mobilenetv2_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds",
    "mobilenetv2_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ",
    "mobilenetv2_x1_4_market1501.pt": "https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5",
    "mobilenetv2_x1_4_dukemtmcreid.pt": "https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN",
    "mobilenetv2_x1_4_msmt17.pt": "https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz",
    "osnet_x1_0_market1501.pt": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x0_75_market1501.pt": "https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",
    "osnet_x0_75_dukemtmcreid.pt": "https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x0_5_market1501.pt": "https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT",
    "osnet_x0_5_dukemtmcreid.pt": "https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_25_market1501.pt": "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",
    "osnet_x0_25_dukemtmcreid.pt": "https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    ####### market1501 models ##################################################
    "resnet50_msmt17.pt": "https://drive.google.com/uc?id=1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf",
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1fhjSS_7SUGCioIf2SWXaRGPqIY9j7-uw",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e",
    "osnet_ibn_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ",
    "osnet_ain_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
}


def show_downloadeable_models():
    print("\nAvailable .pt ReID models for automatic download")
    print(list(__trained_urls.keys()))


def get_model_url(model):

    if isinstance(model, (str, Path)):
        key=model
    elif hasattr(model, 'name'):
        key=model.name
    else:
        key = None
    if key is not None and key in __trained_urls:
        return __trained_urls[key]        
        return __trained_urls[model.name]
    else:
        return None


def is_model_in_model_types(model):
    if model.name in __model_types:
        return True
    else:
        return False


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
    from six.moves import urllib

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
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
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
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(weight_path)
        )
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".format(discarded_layers)
            )







How to Install MongoDB on Windows

    Download the installer:
    Go to the official MongoDB Community Server download page:
    https://www.mongodb.com/try/download/community

    Choose Windows as your OS and download the MSI installer.

    Run the installer:

        Double-click the downloaded .msi file.

        Follow the setup wizard steps.

        Select the Complete installation option.

        Check Install MongoDB as a Service to have MongoDB start automatically with Windows (recommended).

    MongoDB default installation path:
    MongoDB is usually installed under:
    C:\Program Files\MongoDB\Server\6.0\
    The mongosh executable is located in the bin folder inside that directory, e.g.:
    C:\Program Files\MongoDB\Server\6.0\bin

    Add MongoDB to your system PATH (optional but convenient):

        Open Control Panel → System → Advanced system settings → Environment Variables.

        Edit the Path variable and add the path to the MongoDB bin folder, for example:
        C:\Program Files\MongoDB\Server\6.0\bin





    Start using MongoDB:

        If installed as a service, MongoDB runs automatically.

        Open Command Prompt (cmd) and type:

mongosh




How to Install MongoDB on Ubuntu

    Import the MongoDB public GPG key:

wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

    Create a MongoDB source list file:
    Replace focal with your Ubuntu codename if needed (jammy for 22.04, etc.).

echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

    Update package lists:

sudo apt-get update

    Install MongoDB packages:

sudo apt-get install -y mongodb-org

    Start MongoDB service:

sudo systemctl start mongod

    Enable MongoDB to start on boot:

sudo systemctl enable mongod

    Check MongoDB service status:

sudo systemctl status mongod

    Access the MongoDB shell:

mongosh

Let me know if you want the commands adapted to your Ubuntu version or help configuring MongoDB!





****to open the MongoDB shell.

*****Connect to MongoDB shell

mongosh


show dbs


use robotDB


show collections

db.profiles.find().pretty()

//////////////////////////////////////////////////////////////////////////////

Installing ROS 2 Humble + Gazebo Harmonic
1️⃣ Install ROS 2 Humble on Ubuntu 22.04
ROS 2 Humble Ubuntu Installation Guide
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html


🛠️ Installing Gazebo Harmonic on Ubuntu 22.04 (Jammy)

https://gazebosim.org/docs/harmonic/install_ubuntu/


3️⃣ Install Dependent ROS 2 Packages

Open a terminal (Ctrl+Alt+T) and install the necessary packages:

Gazebo:

sudo apt install ros-humble-gazebo-*

Cartographer:

sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros

Navigation2:

sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup

4️⃣ Install TurtleBot3 Packages

Set up the workspace for TurtleBot3 packages:


source /opt/ros/humble/setup.bash
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src

# Clone required repositories
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git

# Install build tools
sudo apt install python3-colcon-common-extensions

# Build the workspace
cd ~/turtlebot3_ws
colcon build --symlink-install

# Source workspace setup automatically
echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc

5️⃣ Configure the Environment

Set up the ROS 2 and Gazebo environment variables for your Remote PC:

echo 'export ROS_DOMAIN_ID=30 # TURTLEBOT3' >> ~/.bashrc
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc



2. Clone the PersonFollowingRobot Repository

   
cd ~/turtlebot3_ws/src
git clone https://github.com/rymmhadh/PersonFollowingRobot.git

3. Copy Required Packages


  Move the required packages into the src folder of your workspace:

   
cd ~/turtlebot3_ws/src/PersonFollowingRobot

# Copy the packages into workspace src
cp -r person_following_pkg ~/turtlebot3_ws/src/
cp -r my_robot_launch ~/turtlebot3_ws/src/


4. Build the Workspace

   
cd ~/turtlebot3_ws/
colcon build --symlink-install
source install/setup.bash

5. Verify Installation
   
ros2 pkg list | grep person_following_pkg
ros2 pkg list | grep my_robot_launch


4. Copy the Python Virtual Environment

To ensure that all Python dependencies are available, copy the .venv-py39 virtual environment into the person_following_pkg directory:

cd ~/turtlebot3_ws/src/PersonFollowingRobot
cp -r .venv-py39 ~/turtlebot3_ws/src/person_following_pkg/

Activate the environment when needed:

cd ~/turtlebot3_ws/src/person_following_pkg/.venv-py39
source bin/activate

5. Build the Workspace

cd ~/turtlebot3_ws/
colcon build --symlink-install
source install/setup.bash






































































#   P e r s o n F o l l o w i n g R o b o t 
 
 







