#!/usr/bin/env python3
"""Evaluate upper-bound models on MNIST subset and challenge dataset.

Supports two backends:
    cnn       -- Train a simple CNN on full MNIST (60K). True upper bound.
    vision    -- Local vision LLM (Qwen2-VL-2B). Multi-modal comparison.

Usage:
    uv run python scripts/evaluate_upper_bound.py --backend cnn
    uv run python scripts/evaluate_upper_bound.py --backend vision
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
from typing import Literal

import numpy as np
import scipy.io as sio
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
except ImportError:
    print(
        "Error: PyTorch not installed. This script is optional and not part of the core project.\n"
        "To install the upper-bound dependencies, run:\n"
        "  uv sync --extra upperbound\n"
        "Or:\n"
        "  uv pip install torch torchvision"
    )
    sys.exit(1)

from src.data_loader import load_digits_data, get_train_test_split, evaluate_accuracy
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir

MNIST_MAT = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
CHALLENGE_MAT = Path(__file__).parent.parent / "data" / "raw" / "challenge" / "cdigits.mat"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# ---------------------------------------------------------------------------
# CNN Backend
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """LeNet-style CNN for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNUpperBound:
    """Train a CNN on full MNIST (60K) and evaluate on target data."""

    def __init__(self, epochs: int = 5, batch_size: int = 128) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CNN device: {self.device}")
        self.model = SimpleCNN().to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
        )
        self._trained = False

    def _train(self) -> None:
        """Train on full MNIST (60K training images)."""
        print("Downloading full MNIST and training CNN...")
        dataset = datasets.MNIST(
            root="/tmp/mnist_data", train=True, download=True, transform=self.transform
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch + 1}/{self.epochs}, loss: {total_loss / len(loader):.4f}")

        self._validate()

    def _validate(self) -> None:
        """Quick validation on the official MNIST test set (10K)."""
        test_dataset = datasets.MNIST(
            root="/tmp/mnist_data", train=False, download=True, transform=self.transform
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        acc = correct / len(test_dataset)
        print(f"  Full MNIST test accuracy: {acc:.4f}")

    def _prepare_batch(self, vectors: np.ndarray) -> torch.Tensor:
        """Convert a batch of 784-dim vectors to a normalized tensor."""
        imgs = vectors.reshape(-1, 28, 28, order="F").astype(np.float32)
        imgs = (imgs / 255.0 - MNIST_MEAN) / MNIST_STD
        return torch.from_numpy(imgs).unsqueeze(1).to(self.device)

    def predict_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of 784-dim vectors."""
        if not self._trained:
            self._train()
            self._trained = True

        self.model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for i in range(0, len(vectors), self.batch_size):
                batch = self._prepare_batch(vectors[i : i + self.batch_size])
                output = self.model(batch)
                preds.extend(output.argmax(dim=1).cpu().tolist())
        return np.array(preds)


# ---------------------------------------------------------------------------
# Vision LLM Backend (Qwen2-VL)
# ---------------------------------------------------------------------------


class VisionLLMUpperBound:
    """Local vision LLM for digit classification."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct") -> None:
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            print(
                "Error: transformers not installed. Run:\n"
                "  uv pip install transformers"
            )
            sys.exit(1)

        print(f"Loading vision model: {self.model_name}")
        print("  (First run downloads ~3GB of weights)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"  Model loaded on {self.device}")

    @staticmethod
    def _prepare_image(vec: np.ndarray) -> Image.Image:
        """Convert 784-dim vector to PIL Image."""
        img_array = vec.reshape((28, 28), order="F").astype(np.uint8)
        return Image.fromarray(img_array).convert("RGB")

    def predict_single(self, image: Image.Image) -> int:
        """Classify a single image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "This is a 28x28 grayscale image of a handwritten digit (0-9). "
                            "Respond with ONLY the digit number, nothing else."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=5, do_sample=False, temperature=None
            )

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        digits = [c for c in response if c.isdigit()]
        return int(digits[0]) if digits else -1

    def predict_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of 784-dim vectors."""
        preds = []
        for i, vec in enumerate(vectors):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(vectors)}...")
            img = self._prepare_image(vec)
            pred = self.predict_single(img)
            preds.append(pred)
        return np.array(preds)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _evaluate_mnist(model) -> dict:
    """Evaluate model on both MNIST trials."""
    if not MNIST_MAT.exists():
        print(f"\nMNIST data not found: {MNIST_MAT}")
        return {}

    print("\n--- MNIST Subset (4,000 samples) ---")
    data = load_digits_data(str(MNIST_MAT))
    results = {}

    for trial in [0, 1]:
        _, X_test, _, y_test = get_train_test_split(data, trial)
        print(f"\nTrial {trial + 1}: {len(y_test)} test samples")

        start = time.time()
        y_pred = model.predict_batch(X_test)
        elapsed = time.time() - start

        accuracy = evaluate_accuracy(y_test, y_pred)
        results[f"trial_{trial + 1}"] = {
            "accuracy": float(accuracy),
            "n_samples": len(y_test),
            "time_sec": elapsed,
        }
        print(f"  Accuracy: {accuracy:.4f} ({elapsed:.1f}s)")

    t1 = results["trial_1"]["accuracy"]
    t2 = results["trial_2"]["accuracy"]
    results["mean"] = float(np.mean([t1, t2]))
    results["std"] = float(np.std([t1, t2]))
    print(f"\n  Mean: {_format_mean_std(results['mean'], results['std'])}")
    return results


def _evaluate_challenge(model, backend_name: str) -> dict:
    """Evaluate model on the challenge dataset."""
    if not CHALLENGE_MAT.exists():
        print(f"\nChallenge data not found: {CHALLENGE_MAT}")
        return {}

    print("\n--- Challenge Dataset (150 samples) ---")
    ch_data = sio.loadmat(CHALLENGE_MAT)
    X_challenge = ch_data["cdigits_vec"].T
    y_challenge = ch_data["cdigits_labels"].flatten()

    trial_accs = []
    for trial in [0, 1]:
        label = "CNN (full MNIST)" if backend_name == "cnn" else "Vision LLM"
        print(f"\n  Trial {trial + 1} — {label}")

        start = time.time()
        y_pred = model.predict_batch(X_challenge)
        elapsed = time.time() - start

        acc = evaluate_accuracy(y_challenge, y_pred)
        trial_accs.append(acc)
        print(f"    Challenge accuracy: {acc:.4f} ({elapsed:.1f}s)")

    return {
        "trial_1": float(trial_accs[0]),
        "trial_2": float(trial_accs[1]),
        "mean": float(np.mean(trial_accs)),
        "std": float(np.std(trial_accs)),
    }


def evaluate_backend(backend_name: Literal["cnn", "vision"], output_dir: Path) -> dict:
    """Run upper-bound evaluation with chosen backend."""
    print("=" * 80)
    print(f"UPPER BOUND EVALUATION — Backend: {backend_name.upper()}")
    print("=" * 80)

    model = CNNUpperBound(epochs=5) if backend_name == "cnn" else VisionLLMUpperBound()

    results = {
        "backend": backend_name,
        "mnist": _evaluate_mnist(model),
        "challenge": _evaluate_challenge(model, backend_name),
    }

    # Save results
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path = logs_dir / f"upper_bound_{backend_name}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if "mean" in results["mnist"]:
        mnist = results["mnist"]
        print(f"  MNIST:     {_format_mean_std(mnist['mean'], mnist['std'])}")
    if "mean" in results.get("challenge", {}):
        ch = results["challenge"]
        print(f"  Challenge: {_format_mean_std(ch['mean'], ch['std'])}")
    print("=" * 80)

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate upper-bound models on MNIST and challenge datasets."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cnn", "vision"],
        default="cnn",
        help="Model backend: cnn (fast, true upper bound) or vision (Qwen2-VL, slower)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Result directory (auto-generated if omitted)",
    )
    args = parser.parse_args()

    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        evaluate_backend(args.backend, output_dir)


if __name__ == "__main__":
    main()
