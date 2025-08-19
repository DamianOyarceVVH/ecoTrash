from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    photo_filename = "latest.jpg"

    if request.method == "POST":
        file = request.files.get('photo')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
            file.save(filepath)
            return redirect(url_for('index'))

    material_identificado = "Esperando..."
    confidence = "--"

    return render_template(
        "index.html",
        photo_file=photo_filename,
        material_identificado=material_identificado,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
