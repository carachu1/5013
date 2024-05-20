from model import *

from datasets import *
from torchvision import transforms
from torchvision.transforms import v2

transform_train = transforms.Compose([
    transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = load_dataset("imagenet-1k")
train_dataset = dataset['train']
test_dataset = dataset['test']
print(train_dataset)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        transform_train(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [transform_test(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

train_dataset.set_transform(preprocess_train)
test_dataset.set_transform(preprocess_val)

from transformers import AutoImageProcessor,  Trainer, TrainingArguments,EfficientNetImageProcessor, AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b4")
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b4")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4096,
    per_device_eval_batch_size=4096,
    num_train_epochs=400,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Define metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).astype(float).mean().item()}

model = effnetv2_s(num_classes=1000)
mixup = v2.MixUp(num_classes=1000, alpha=0.2)

def collate_fn(examples):
    def default_collate(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    return mixup(*default_collate(examples))

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# Train the model
trainer.train()