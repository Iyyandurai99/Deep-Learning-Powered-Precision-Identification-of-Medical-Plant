import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load pre-trained Keras model for species prediction
try:
    model_species = load_model('leaf.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
