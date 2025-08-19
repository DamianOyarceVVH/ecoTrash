from flask import Flask, render_template, request
import os
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/fotos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['imageBase64']
    header, encoded = data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    filename = os.path.join(UPLOAD_FOLDER, 'foto.jpg')
    with open(filename, "wb") as f:
        f.write(binary_data)

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
