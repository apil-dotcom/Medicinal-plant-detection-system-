from flask import Flask, render_template, request, jsonify
import os
import uuid
from uuid import uuid4

from utils.predict import predict_image
from utils.chat import chat_with_ai
from utils.translations import translations
from flask import session
print("Current Working Directory:", os.getcwd())


app = Flask(__name__)
app.secret_key = "medicinal_plant_secret"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    lang = request.args.get("lang", session.get("lang", "en"))
    session["lang"] = lang
    return render_template("index.html", t=translations[lang], lang=lang)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    plant_name = None
    confidence = None
    image_path = None
    benefits = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            plant_name, confidence, benefits = predict_image(image_path)
            prediction = plant_name

    # Generate UUID for cache-busting
    cache_buster = uuid4().hex

    return render_template(
        "predict.html",
        prediction=prediction,
        plant_name=plant_name,
        confidence=confidence,
        image_path=image_path,      # pass the image path
        benefits=benefits,
        cache_buster=cache_buster   # pass the UUID to template
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    plant_name = data.get("plant_name", "")

    if not question:
        return jsonify({"answer": "Please ask a question"})

    answer = chat_with_ai(question, plant_name)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
