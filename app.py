from flask import Flask, request, jsonify, send_file
import torch
from PIL import Image
from torchvision import transforms
import io

app = Flask(__name__)

# Load your model
model = torch.load('best_metric_model.pt')
model.eval()

# Define transformation for the input image
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream)
    image = preprocess(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
    
    # Here, you should process the output from your model.
    # This is just a placeholder. Replace it with your actual result processing.
    result_text = "Detected object or result here" 

    # If the model produces an image, you can save it and return the path.
    # For this example, we are just returning text results.
    
    return jsonify({'result': result_text})

if __name__ == '__main__':
    app.run(debug=True)
