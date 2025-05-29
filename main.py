import os
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Plant information dictionary
plants_info = {
    "Alpinia Galanga": {
        "Name": "Rasna",
        "Use": "Anti-inflammatory, digestive aid.",
        "Cure": "Treats arthritis, rheumatism, and indigestion."
    },
    "Amaranthus Viridis": {
        "Name": "Arive-Dantu",
        "Use": "Diuretic, anti-inflammatory.",
        "Cure": "Treats ulcers, wounds, and fever. "
    },
    "Artocarpus Heterophyllus": {
        "Name": "Jackfruit",
        "Use": "Antioxidant, immune booster.",
        "Cure": "Helps with digestive issues, strengthens immunity."
    },
    "Azadirachta Indica": {
        "Name": "Neem",
        "Use": "Antibacterial, antiviral, antifungal.",
        "Cure": "Treats skin diseases, infections, and purifies blood."
    },
    "Basella Alba": {
        "Name": "Basale",
        "Use": "Rich in vitamins, anti-inflammatory.",
        "Cure": "Treats anemia, skin problems, and boosts immunity."
    },
    "Brassica Juncea": {
        "Name": "Indian Mustard",
        "Use": "Antibacterial, detoxifying.",
        "Cure": "Treats cold, arthritis, and stimulates appetite."
    },
    "Carissa Carandas": {
        "Name": "Karanda",
        "Use": "Digestive aid, blood purifier.",
        "Cure": "Helps with indigestion, diarrhea, and skin infections."
    },
    "Citrus Limon": {
        "Name": "Lemon",
        "Use": "Antioxidant, digestive, detoxifying.",
        "Cure": "Treats colds, indigestion, and kidney stones."
    },
    "Ficus Auriculata": {
        "Name": "Roxburgh Fig",
        "Use": "Antidiabetic, antioxidant.",
        "Cure": "Manages diabetes, treats digestive issues. "
    },
    "Ficus Religiosa": {
        "Name": "Peepal Tree",
        "Use": "Antioxidant, anti-inflammatory.",
        "Cure": "Treats asthma, skin diseases, and diabetes."
    },
    "Hibiscus Rosa-sinensis": {
        "Name": "Hibiscus",
        "Use": "Antioxidant, anti-inflammatory.",
        "Cure": "Lowers blood pressure, promotes hair growth, treats skin issues."
    },
    "Jasminum": {
        "Name": "Jasmine",
        "Use": "Antidepressant, sedative.",
        "Cure": "Reduces stress, insomnia, and anxiety."
    },
    "Mangifera Indica": {
        "Name": "Mango",
        "Use": "Antioxidant, digestive.",
        "Cure": "Treats diarrhea, boosts immunity, promotes digestion. "
    },
    "Mentha": {
        "Name": "Mint",
        "Use": "Digestive aid, cooling.",
        "Cure": "Relieves indigestion, nausea, and respiratory issues."
    },
    "Moringa Oleifera": {
        "Name": "Drumstick",
        "Use": "Rich in vitamins, anti-inflammatory.",
        "Cure": "Treats malnutrition, boosts immunity, and reduces inflammation."
    },
    "Muntingia Calabura": {
        "Name": "Jamaica Cherry",
        "Use": "Antioxidant, anti-inflammatory.",
        "Cure": "Treats headaches, reduces blood sugar, and alleviates respiratory issues."
    },
    "Murraya Koenigii": {
        "Name": "Curry Leaves",
        "Use": "Antioxidant, antimicrobial.",
        "Cure": "Helps with diabetes, indigestion, and hair problems."
    },
    "Nerium Oleander": {
        "Name": "Oleander",
        "Use": "Cardiotonic (in low doses), anti-inflammatory.",
        "Cure": "Used cautiously for heart conditions, but toxic in high doses."
    },
    "Nyctanthes Arbor-tristis": {
        "Name": "Parijata",
        "Use": "Antipyretic, anti-inflammatory.",
        "Cure": "Treats fever, arthritis, and malaria. "
    },
    "Ocimum Tenuiflorum": {
        "Name": "Tulsi",
        "Use": "Antiviral, antibacterial, adaptogen.",
        "Cure": "Treats colds, stress, and respiratory issues."
    },
    "Piper Betle": {
        "Name": "Betel Leaf",
        "Use": "Antimicrobial, digestive.",
        "Cure": "Treats coughs, indigestion, and oral health issues."
    },
    "Plectranthus Amboinicus": {
        "Name": "Mexican Mint",
        "Use": "Antitussive, digestive.",
        "Cure": "Treats colds, coughs, and indigestion. "
    },
    "Pongamia Pinnata": {
        "Name": "Indian Beech",
        "Use": "Anti-inflammatory, wound healing.",
        "Cure": "Treats skin diseases, wounds, and inflammation."
    },
    "Psidium Guajava": {
        "Name": "Guava",
        "Use": "Antidiabetic, antioxidant.",
        "Cure": "Controls blood sugar, improves digestion, and boosts immunity. "
    },
    "Punica Granatum": {
        "Name": "Pomegranate",
        "Use": "Antioxidant, anti-inflammatory.",
        "Cure": "Treats heart disease, diabetes, and digestive issues. "
    },
    "Santalum Album": {
        "Name": "Sandalwood",
        "Use": "Cooling, antiseptic.",
        "Cure": "Treats skin problems, reduces stress, and treats urinary infections."
    },
    "Syzygium Cumini": {
        "Name": "Jamun",
        "Use": "Antidiabetic, antioxidant.",
        "Cure": "Controls diabetes, improves digestion, and boosts immunity."
    },
    "Syzygium Jambos": {
        "Name": "Rose Apple",
        "Use": "Antioxidant, digestive aid.",
        "Cure": "Treats diarrhea, controls blood sugar, and aids in digestion."
    },
    "Tabernaemontana Divaricata": {
        "Name": "Crape Jasmine",
        "Use": "Antidiabetic, anti-inflammatory.",
        "Cure": "Treats diabetes, headaches, and skin problems."
    },
    "Trigonella Foenum-graecum": {
        "Name": "Fenugreek",
        "Use": "Antidiabetic, digestive aid.",
        "Cure": "Helps with diabetes, indigestion, and cholesterol levels."
    }
}



# Load pre-trained Keras model for species prediction
model_species = load_model('leaf.h5')

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Image processing function
def process_image(input_data):
    if isinstance(input_data, str):  # If the input is a file path (string)
        img = image.load_img(input_data, target_size=(256, 256))
    else:  # If the input is an OpenCV frame (numpy array)
        img = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (256, 256))  # Resize the image to (256, 256)
        img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0) / 255.0  # Normalizing the image
    return img

# Predict plant species
def predict_crop_species(img):
    predictions = model_species.predict(img)
    predicted_index = np.argmax(predictions)
    class_labels = list(plants_info.keys())
    return class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

frame = None
# OpenCV video stream function
def gen_frames():
    global frame
    camera = cv2.VideoCapture(0)  # Open webcam (0 is the default camera)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame for streaming over HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
@app.route('/home')
def home():
    return render_template('home_page.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['GET'])
def capture_image():
    # Open webcam
    cap = cv2.VideoCapture(0)
    ret, captured_frame = cap.read()
    cap.release()

    if not ret:
        return "Failed to capture image", 400

    # Define the directory to save captured images
    img_directory = 'captured_images'
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    # Save the captured image in the specified directory
    img_filename = os.path.join(img_directory, 'captured_frame.jpg')
    cv2.imwrite(img_filename, captured_frame)

    # Preprocess the captured frame
    processed_img = process_image(captured_frame)

    # Make prediction
    prediction = predict_crop_species(processed_img)
    plant_description = plants_info.get(prediction, {})

    return render_template(
        'capture_prediction.html',
        prediction=prediction,
        plant_description=plant_description,
        uploaded_image=img_filename
    )


# Routes for rendering HTML templates
@app.route('/symptoms')
def symptoms():
    """List symptoms and provide links to treatment details."""
    symptoms_list = ["Cold", "Cough", "Fever", "Headache", "Skin Irritation"]
    return render_template('symptoms.html', symptoms=symptoms_list)


@app.route('/symptom_detail/<symptom>')
def symptom_detail(symptom):
    """Provide treatment details for a specific symptom."""
    remedies = {
        "Cough": {
            "plant": "Tulsi",
            "quantity": "5-10 leaves",
            "preparation": "Take 5-7 fresh Tulsi leaves and wash them thoroughly."
                            "Boil the leaves in a cup of water for about 10 minutes."
                            "Optionally, add a small piece of ginger and a teaspoon of honey to enhance the effect and taste.Strain the mixture and allow it to cool slightly.Drink this herbal tea twice daily to soothe the throat and relieve coughing symptoms.",
            "Description" : "Tulsi is known for its anti-inflammatory, antiviral, and antimicrobial properties, which can help relieve respiratory issues like coughs. It helps to loosen mucus, reduce throat irritation, and improve overall respiratory health."
        },
        "Cold": {
            "plant": "Peppermint",
            "quantity": "5-7 fresh leaves",
            "preparation": "Take 5-10 fresh peppermint leaves and wash them thoroughly."
                            "Boil the leaves in a cup of water for about 10 minutes to make a tea. Add a teaspoon of honey for additional soothing benefits."
                            "Strain the mixture and sip while warm.",
            "Description" : "Peppermint contains menthol, which has natural decongestant properties that help open up the airways, making it easier to breathe. The menthol soothes a sore throat and helps reduce nasal congestion, providing relief from cold symptoms. Drinking peppermint tea can also help reduce body aches and promote relaxation."
        },
        "Fever": {
            "plant": "Papaya Leaf ",
            "quantity": "1 fresh leaves",
            "preparation": "Take a fresh papaya leaf and wash it thoroughly."
                            "Crush the leaf to extract the juice. You can do this by pounding it gently in a mortar or using a blender."
                            "Strain the juice into a cup. (You should aim to get about 1-2 tablespoons of juice.)"
                            "If desired, add a small amount of honey to improve taste.",
            "Description" : "Papaya leaf is well-known for its ability to boost platelet count and support immune function, especially in cases of fever due to viral infections like dengue. The leaf contains powerful antioxidants and enzymes, including papain, which aid digestion, reduce inflammation, and help the body recover. Papaya leaf juice is also rich in vitamins A, C, and E, which strengthen the immune system and support the body in fighting off fever and infection.",
        },
        "Headache": {
            "plant": "Basil",
            "quantity": "6-8 fresh leaves",
            "preparation": "Take 6-8 fresh basil leaves and wash them well."
                            "Boil the basil leaves in a cup of water for about 5-10 minutes to prepare a tea."
                            "Optionally, add a small piece of ginger or a teaspoon of honey for added benefits."
                            "Strain and drink the tea while it’s warm.",
            "Description": "Basil has analgesic and muscle-relaxant properties that can ease headache symptoms, especially those caused by muscle tension or stress. Drinking basil tea can help soothe the body and relieve tension, often bringing comfort from headaches. Inhaling the steam from the basil-infused water can also provide additional relief by relaxing the muscles around the head and neck."
        },
        "Skin Irritation": {
            "plant": "Aloe Vera",
            "quantity": "Fresh gel 1 - 2 tablespoon",
            "preparation": "Take a fresh aloe vera leaf and cut it open to extract the gel."
                            "Scoop out the clear gel from inside the leaf with a spoon."
                            "Apply a thin layer of the gel directly to the irritated area of the skin."
                            "Let it sit on the skin for 15–20 minutes, then rinse off with lukewarm water.",
            "Description" : "Aloe vera gel is highly regarded for its soothing and anti-inflammatory properties, making it ideal for treating skin irritation, redness, and minor burns. The gel contains compounds like polysaccharides, which promote skin healing and regeneration, and vitamins C and E, which nourish the skin. Aloe vera also has antibacterial and antifungal properties, providing relief from itching, cooling the skin, and helping reduce any swelling or discomfort associated with skin irritation."
        }
    }

    remedy = remedies.get(symptom)
    return render_template('symptom_detail.html', symptom=symptom, remedy=remedy)


@app.route('/upload')
def upload():
    """Render the upload page for image-based predictions."""
    return render_template('upload.html')


@app.route('/plant_species', methods=['POST'])
def prediction():
    """Handle file upload and predict the plant species."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = process_image(file_path)
        prediction = predict_crop_species(img)
        plant_description = plants_info.get(prediction, {})

        return render_template('prediction.html', prediction=prediction, plant_description=plant_description,
                               uploaded_image=filename)
    else:
        return jsonify({'error': 'File type not allowed'}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
