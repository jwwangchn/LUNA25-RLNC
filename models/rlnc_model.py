import copy
import torch
from torch import nn
from torchvision.models.video import r3d_18


class RlncModelV1(nn.Module):
    def __init__(self, classes=1, pretrained=True):
        super(RlncModelV1, self).__init__()
        self.resnet3d = r3d_18(pretrained=pretrained)
        self._modify_layer()
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, classes)

    def _modify_layer(self):
        original = self.resnet3d.stem[0]
        adjusted = nn.Conv3d(
            in_channels=1,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=(original.bias is not None)
        )
        if original.weight.size(1) == 3:
            adjusted.weight.data = original.weight.mean(dim=1, keepdim=True)
        if original.bias is not None:
            adjusted.bias.data = original.bias.data
        self.resnet3d.stem[0] = adjusted

    def forward(self, input_tensor):
        return self.resnet3d(input_tensor)


class RlncModelV2(nn.Module):
    def __init__(self, classes=1, pretrained=True):
        super(RlncModelV2, self).__init__()
        self.resnet3d = r3d_18(pretrained=pretrained)
        self._modify_layer()
        self.resnet3d.fc = nn.Sequential(nn.Dropout(0), nn.Linear(self.resnet3d.fc.in_features, classes))

    def _modify_layer(self):
        original = self.resnet3d.stem[0]
        adjusted = nn.Conv3d(
            in_channels=1,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=(original.bias is not None)
        )
        if original.weight.size(1) == 3:
            adjusted.weight.data = original.weight.mean(dim=1, keepdim=True)
        if original.bias is not None:
            adjusted.bias.data = original.bias.data
        self.resnet3d.stem[0] = adjusted

    def forward(self, input_tensor):
        return self.resnet3d(input_tensor)


class ModelEnsembler:
    def __init__(self, model_list):
        self.models = model_list

    def __call__(self, input_data):
        with torch.no_grad():
            predictions = [model(input_data) for model in self.models]
            return torch.stack(predictions).mean(dim=0)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()


class MultiInferencer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, input_batch):
        target_device = self.device if self.device else input_batch.device
        num_samples = input_batch.size(0)
        augmented_data = []

        for idx in range(num_samples):
            sample = input_batch[idx:idx + 1]
            augmentations = self._augment(sample)
            augmented_data.extend(augmentations)

        concatenated = torch.cat(augmented_data, dim=0)
        predictions = []

        with torch.no_grad():
            for start_idx in range(0, concatenated.size(0), 32):
                end_idx = start_idx + 32
                batch_segment = concatenated[start_idx:end_idx].to(target_device)
                outputs = self.model(batch_segment)
                predictions.append(outputs.cpu())

        all_predictions = torch.cat(predictions, dim=0)
        aug_per_sample = len(self._augment(input_batch[0:1]))
        reshaped_preds = all_predictions.view(num_samples, aug_per_sample, -1)
        averaged_logits = reshaped_preds.mean(dim=1)

        return averaged_logits.to(target_device)

    def _augment(self, tensor):
        transformations = [tensor]

        for dim in [2, 3, 4]:
            flipped = torch.flip(tensor, dims=[dim])
            transformations.append(flipped)

        rotations = [
            torch.rot90(tensor, k=1, dims=[2, 3]),
            torch.rot90(tensor, k=1, dims=[2, 4]),
            torch.rot90(tensor, k=1, dims=[3, 4])
        ]
        transformations.extend(rotations)
        transformations += [tensor * scale for scale in [0.9, 1.1]]
        for std in [0.01, 0.02]:
            noise = torch.randn_like(tensor) * std
            transformations.append(tensor + noise)

        unique_transforms = self.duplicates(transformations)
        return unique_transforms

    def duplicates(self, tensors):
        unique_set = set()
        unique_list = []

        for item in tensors:
            tensor_bytes = item.cpu().numpy().tobytes()
            tensor_hash = hash(tensor_bytes)
            if tensor_hash not in unique_set:
                unique_set.add(tensor_hash)
                unique_list.append(item)

        return unique_list