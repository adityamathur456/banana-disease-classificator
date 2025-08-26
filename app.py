import gradio as gr
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('banana_disease_densenet121.keras')

# Load class names
class_names = np.load("class_names.npy", allow_pickle=True)

# Preprocess the input image
def preprocess_image(img):
    if img is None:
        return None
    img = img.resize((256, 256))  # Resize to match training
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def predict_disease(img):
    img = preprocess_image(img)
    if img is None:
        return "⚠️ No image provided", None
    predictions = model.predict(img)[0]
    predicted_class = np.argmax(predictions)
    confidence_scores = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return f"Predicted: {class_names[predicted_class]}", confidence_scores


# 🌿 Custom theme
custom_theme = gr.themes.Soft()

# UI
with gr.Blocks(theme=custom_theme, css="""
#title {
  font-size: 32px;
  font-weight: bold;
  text-align: center;
  color: white;
  animation: fadeIn 2s ease-in-out;
}
#subtitle {
  text-align: center;
  font-size: 16px;
  color: #ccc;
  animation: fadeInUp 2s ease-in-out;
}
#prediction_box {
  font-size: 18px;
  padding: 12px;
  border-radius: 8px;
  min-height: 60px;   /* ensures readable area */
  word-wrap: break-word;
  white-space: normal;
}
.gr-button {
  transition: 0.3s;
}
.gr-button:hover {
  transform: scale(1.05);
  box-shadow: 0px 4px 20px rgba(0, 128, 0, 0.4);
}
@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}
@keyframes fadeInUp {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
""") as demo:

    gr.Markdown("<div id='title'>Banana Leaf Disease Classifier</div>")
    gr.Markdown("<div id='subtitle'>Upload a banana leaf image, and our AI will diagnose the disease</div>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", image_mode="RGB", label="Upload Banana Leaf")
            predict_btn = gr.Button("🔍 Predict Disease", variant="primary", elem_id="predict_btn")
        
        with gr.Column(scale=1):
            # ✅ Changed from Textbox to Text
            output_label = gr.Text(label="Prediction", interactive=False, elem_id="prediction_box")
            with gr.Accordion("Confidence Scores", open=True):
                output_conf = gr.Label(label="Scores")

    predict_btn.click(fn=predict_disease, inputs=image_input, outputs=[output_label, output_conf])

# Launch app
demo.launch(share=True)
