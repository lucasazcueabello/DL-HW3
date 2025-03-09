from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.c3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.gelu = torch.nn.GELU()
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
            self.dropout = torch.nn.Dropout(p=0.1)

            if in_channels != out_channels or stride != 1:
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            else:
                self.residual = nn.Identity()

        def forward(self, x):
            x1 = self.dropout(self.gelu(self.batch_norm(self.c1(x))))
            x1 = self.dropout(self.gelu(self.batch_norm(self.c2(x1))))
            x1 = self.dropout(self.gelu(self.batch_norm(self.c3(x1))))
            return x1 + self.residual(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        num_blocks: int = 6
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        cnn_layers = [
            #torch.nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3),
            #torch.nn.GELU(),
        ]
        c1 = in_channels
        for _ in range(num_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        logits = self.network(z)

        return logits.squeeze()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)



class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        num_blocks: int = 4,
        depth: int = 4,
        wf: int = 4,
        padding: bool = True,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                DownConvBlock(prev_channels, 2 ** (wf + i), padding, num_blocks, stride=2, useSigmoid=False)
            )
            prev_channels = 2 ** (wf + i)

        prev_channels = 2 * prev_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UpConvBlock(prev_channels, 2 ** (wf + i-1), padding, num_blocks)
            )
            prev_channels = 2 ** (wf + i)

        self.segmentation = nn.Conv2d(int(prev_channels/2), num_classes, kernel_size=1)
        self.estimation = nn.Conv2d(int(prev_channels/2), 1, kernel_size=1)
        #self.maxPool = nn.MaxPool2d(kernel_size=2)

        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        blocks = []
        for i, down in enumerate(self.down_path):
            z = down(z)
            if i != len(self.down_path) - 1:
                blocks.append(z)
                #z = self.maxPool(z)
        

        for i, up in enumerate(self.up_path):
            if i > 0:
                # For layers with skip connections, pass the corresponding bridge tensor
                z = up(z, blocks[-i])  # Pass the bridge tensor
            else:
                # For layers without skip connections, pass `None` or a zeros tensor
                z = up(z, None)  # No bridge tensor


        return self.segmentation(z), self.estimation(z).squeeze()

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth

class DownConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, num_blocks = 1, stride=2, useSigmoid = False):
        super(DownConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=int(padding)))
        if useSigmoid:
            block.append(nn.Sigmoid())
        else:
            block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))
        for i in range(num_blocks-1):
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
            if useSigmoid:
                block.append(nn.Sigmoid())
            else:
                block.append(nn.ReLU())
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    
class UpConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, num_blocks = 1):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        
        self.conv_block = DownConvBlock(out_size, out_size, padding, num_blocks, stride=1, useSigmoid=True)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge=None):
        if bridge is not None:
            # Otherwise, crop the bridge to match the shape of `up`
            crop1 = self.center_crop(bridge, x.shape[2:])
            x = torch.cat([x, crop1], dim=1)
        else:
            crop1 = torch.zeros_like(x)
            x = torch.cat([x, crop1], dim=1)
        out = self.up(x)
        out = self.conv_block(out)
        return out
    

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
