import streamlit as st
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

# Define the model
class Sentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(384, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the trained sentiment model
model2 = Sentiment()
model2.load_state_dict(torch.load("Sentiment_Analysis.pth", map_location=torch.device("cpu")))
model2.eval()

# Streamlit UI
st.title("🗣️ Sentiment Analysis")

user = st.text_area("Enter a review:")

if st.button("Analyze"):
    if user.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        with st.spinner("Analyzing..."):
            # Embedding the input
            trains_embeddings = model.encode(user, show_progress_bar=True)
            input_tensor = torch.tensor(trains_embeddings, dtype=torch.float32)

            # Inference
            with torch.inference_mode():
                pred = model2(input_tensor)
                pred_labels = torch.sigmoid(pred)

            st.subheader("Prediction Result:")
            # st.write(f"Raw Output: {pred.item():.4f}")
            st.write(f"Confidence Rate(sigmoid): {pred_labels.item() * 100:.2f}%")

            # Sentiment logic
            if pred_labels.item() < 0.3:
                st.error("😠 Negative Statement")
            elif 0.3 < pred_labels.item() < 0.7:
                st.info("😐 Neutral Statement")
            else:
                st.success("😊 Positive Statement")
