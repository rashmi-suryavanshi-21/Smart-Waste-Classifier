import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

IMG_SIZE = (224, 224)
THRESHOLD = 0.8   # isse kam confidence = "unsure"

# Alphabetical order, bilkul training jaisa (image_dataset_from_directory ka order)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
RECYCLABLE = {"cardboard": True, "glass": True, "metal": True,
              "paper": True, "plastic": True, "trash": False}

# cache_resource: model sirf ek baar load hota hai, har click pe nahi
@st.cache_resource
def load_model():
    return keras.models.load_model("models/mobilenetv2.keras")

def predict(image, model):
    # Model ke andar Rescaling layer hai, to yahan normalize nahi karna
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")[np.newaxis, ...]   # shape (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]
    return probs

st.title("Smart Waste Classifier")
st.write("Upload a photo or use your webcam to classify waste (SDG 11 and 12).")

model = load_model()

# Radio se user chunta hai. Camera widget sirf "Use webcam" chunne par banta hai,
# isliye upload karte waqt camera on nahi hoga.
mode = st.radio("Choose input", ["Upload image", "Use webcam"], horizontal=True)
image = None
if mode == "Upload image":
    f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if f:
        image = Image.open(f)
else:
    c = st.camera_input("Take a photo")
    if c:
        image = Image.open(c)

if image is not None:
    st.image(image, caption="Your image", width=350)
    probs = predict(image, model)
    top = int(np.argmax(probs))
    label, conf = CLASS_NAMES[top], float(probs[top])

    if conf < THRESHOLD:
        st.warning(f"Unsure: looks like **{label}** ({conf:.0%}). Please check manually.")
    else:
        st.success(f"Prediction: **{label}** ({conf:.0%})")
    st.write("Recyclable: " + ("Yes" if RECYCLABLE[label] else "No"))

    # Har class ka probability bar chart
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(6)})