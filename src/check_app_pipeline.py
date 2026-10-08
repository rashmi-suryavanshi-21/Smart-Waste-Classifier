from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
model = keras.models.load_model("models/mobilenetv2.keras")

correct = total = 0
for cls in CLASS_NAMES:
    for p in sorted((Path("data/processed/test") / cls).glob("*.*")):
        # bilkul app jaisa preprocessing
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype="float32")[np.newaxis, ...]
        pred = CLASS_NAMES[int(np.argmax(model.predict(arr, verbose=0)[0]))]
        correct += pred == cls
        total += 1
print(f"App-style accuracy on test set: {correct}/{total} = {correct/total:.1%}")