import torchvision.transforms as transforms
import torch


class RandomGrayscaleVariation:
    """
    Apply random variations to grayscale images after conversion.
    This helps the model be robust to different grape colors (red, green, black)
    by adding random brightness/contrast variations in grayscale space.
    """
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.8, 1.2), p=0.8):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor: Grayscale tensor in range [0, 1] or normalized
        Returns:
            tensor with random grayscale variations applied
        """
        if torch.rand(1).item() < self.p:
            # Random brightness adjustment in grayscale
            brightness_factor = torch.empty(1).uniform_(*self.brightness_range).item()
            tensor = tensor * brightness_factor

            # Random contrast adjustment in grayscale
            contrast_factor = torch.empty(1).uniform_(*self.contrast_range).item()
            mean = tensor.mean()
            tensor = (tensor - mean) * contrast_factor + mean

            # Clamp to valid range
            tensor = torch.clamp(tensor, 0.0, 1.0)

        return tensor


def get_train_transforms():
    """
    Extremely robust augmentation pipeline with random variations
    Includes: color, geometric, quality, and perspective transforms
    """
    return transforms.Compose(
        [
            # Aggressive random cropping and scaling (60%-100% of image)
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),

            # Random flipping (50% chance each)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # Full rotation capability
            transforms.RandomRotation(degrees=360),

            # VERY aggressive color augmentation with extreme variations
            transforms.ColorJitter(
                brightness=(0.3, 1.5),  # Much darker (0.3) to brighter (1.5)
                contrast=(0.5, 1.8),    # Low to high contrast
                saturation=(0.4, 1.6),  # Desaturated to oversaturated
                hue=0.2                 # Significant hue shifts
            ),

            # Random perspective transform (warping/twisting) - 40% chance
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),

            # Random affine transforms (translation, shear, scale) - 50% chance
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),  # Random translation up to 15%
                    scale=(0.85, 1.15),      # Random scaling 85%-115%
                    shear=15                 # Random shearing/twisting up to 15°
                )
            ], p=0.5),

            # Random blur (simulate out of focus) - 40% chance
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))
            ], p=0.4),

            # Random sharpness adjustment - 40% chance
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),

            # Random auto contrast - 30% chance
            transforms.RandomAutocontrast(p=0.3),

            # Random equalization (histogram equalization) - 30% chance
            transforms.RandomEqualize(p=0.3),

            # Random posterize (reduce bits - quality degradation) - 20% chance
            transforms.RandomApply([
                transforms.RandomPosterize(bits=4, p=1.0)
            ], p=0.2),

            # Random solarize (invert pixels above threshold) - 10% chance
            transforms.RandomApply([
                transforms.RandomSolarize(threshold=128, p=1.0)
            ], p=0.1),

            # Convert to grayscale (removes color information - works for all grape colors)
            transforms.Grayscale(num_output_channels=1),

            # Convert to tensor
            transforms.ToTensor(),

            # Random grayscale variations (80% chance)
            # This makes the model robust to different grape colors (red, green, black)
            # by varying the grayscale intensity and contrast
            RandomGrayscaleVariation(
                brightness_range=(0.6, 1.4),  # Random gray brightness
                contrast_range=(0.7, 1.3),    # Random gray contrast
                p=0.8
            ),

            # Random erasing (simulate occlusion/missing parts) - 30% chance
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),

            # Normalize
            transforms.Normalize(mean=[0.5], std=[0.5])

        ]
    )


def get_test_transforms():

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
