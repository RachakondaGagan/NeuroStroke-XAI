from PIL import Image

try:
    import torchvision.transforms as transforms
    _HAS_TORCHVISION = True
except ImportError:
    transforms = None
    _HAS_TORCHVISION = False


IMAGE_SIZE = 224


# ============================================================
# Custom aspect-ratio-preserving resize + padding
# ============================================================

class ResizeAndPad:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to IMAGE_SIZE x IMAGE_SIZE.
    """

    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        # Original dimensions
        width, height = image.size

        # Scale image so the larger dimension becomes IMAGE_SIZE
        scale = self.size / max(width, height)

        new_width = round(width * scale)
        new_height = round(height * scale)

        # Preserve aspect ratio
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        # Create square canvas
        padded_image = Image.new(
            image.mode,
            (self.size, self.size),
            self.fill
        )

        # Center image on canvas
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        padded_image.paste(
            image,
            (left, top)
        )

        return padded_image


# ============================================================
# Training transformations
# ============================================================

if _HAS_TORCHVISION:
    train_transform = transforms.Compose([
        # Preserve aspect ratio and pad to 224 x 224
        ResizeAndPad(IMAGE_SIZE),

        # Small clinically conservative geometric augmentation
        transforms.RandomRotation(
            degrees=5,
            fill=0
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            fill=0
        ),

        # Convert PIL image → PyTorch tensor
        transforms.ToTensor(),

        # Temporary ImageNet normalization.
        # We will verify this choice against training-set
        # statistics before finalizing the experiment.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        # Same deterministic resizing/padding
        ResizeAndPad(IMAGE_SIZE),

        # No random augmentation
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
else:
    train_transform = None
    val_test_transform = None

