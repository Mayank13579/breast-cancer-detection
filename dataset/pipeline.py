# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

print(model)
print(processor)