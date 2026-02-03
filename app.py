
from flask import Flask, render_template, request
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os

app = Flask(__name__)

# ---------------------------
# Device and model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 5)
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.eval()

# # ---------------------------
# # Classes (ensure same order as train_dataset.class_to_idx)
# # Option 1: Manually same as train_dataset.class_to_idx
# classes = ['Brown Rust','Healthy','Mildew','Septoria','Yellow Rust']

# remedies = {
#     "Brown Rust": "Use fungicide X, maintain dry conditions...",
#     "Healthy": "No action needed",
#     "Mildew": "Apply fungicide Y, remove affected leaves...",
#     "Septoria": "Remove infected plants, apply fungicide Z...",
#     "Yellow Rust": "Fungicide A, crop rotation..."
# }

# # ---------------------------
# # Image transform with normalization
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# def predict_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         outputs = model(image)
#         _, preds = torch.max(outputs, 1)
#     label = classes[preds.item()]
#     return label, remedies[label]

# # ---------------------------
# @app.route("/", methods=["GET","POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file:
#             file_path = os.path.join("static", file.filename)
#             file.save(file_path)
#             label, remedy = predict_image(file_path)
#             return render_template("index.html", filename=file.filename, label=label, remedy=remedy)
#     return render_template("index.html", filename=None)

# # ---------------------------
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# ---------------------------
# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ---------------------------
# Classes (same order as train_dataset.class_to_idx)
classes = ['Brown Rust','Healthy','Mildew','Septoria','Yellow Rust']

remedies = {
    "Brown Rust": {
         "Use fungicide X, maintain dry conditions, avoid overhead irrigation.",
         "फफूंदनाशी X का उपयोग करें, खेत को सूखा रखें और ऊपर से पानी न डालें।"
    },
    "Healthy": {
         "No action needed. Keep monitoring the crop.",
         "कोई कार्रवाई आवश्यक नहीं। फसल की निगरानी करते रहें।"
    },
    "Mildew": {
         "Apply fungicide Y, remove affected leaves, ensure good air circulation.",
         "फफूंदनाशी Y का छिड़काव करें, प्रभावित पत्तियों को हटा दें, और अच्छी हवा का प्रवाह सुनिश्चित करें।"
    },
    "Septoria": {
         "Remove infected plants, apply fungicide Z, avoid dense planting.",
         "संक्रमित पौधों को हटा दें, फफूंदनाशी Z का उपयोग करें, और घनी बुवाई से बचें।"
    },
    "Yellow Rust": {
        "Use fungicide A, follow crop rotation, avoid excessive nitrogen fertilizer.",
         "फफूंदनाशी A का उपयोग करें, फसल चक्र अपनाएं, और अत्यधिक नाइट्रोजन उर्वरक से बचें।"
    }
}

# ---------------------------
# Image transform with normalization
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    label = classes[preds.item()]
    return label, remedies[label]

# ---------------------------
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            label, remedy = predict_image(file_path)
            return render_template("index.html", filename=file.filename, label=label, remedy=remedy)
    return render_template("index.html", filename=None)

# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)