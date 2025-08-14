from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import threading
import random
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_super_clave_secreta'
socketio = SocketIO(app)

# La ruta principal que sirve el archivo index.html
@app.route('/')
def index():
    return render_template('index.html')

# Lógica que simula la clasificación de la ESP32-CAM
def simular_y_clasificar():
    materiales = [
        {"material": "plastico", "confidence": 95},
        {"material": "papel", "confidence": 88},
        {"material": "organico", "confidence": 92},
        {"material": "metal", "confidence": 76},
    ]

    while True:
        # Simula la espera de una detección del sensor ultrasónico
        time.sleep(5) 
        
        # Simula la petición y el resultado de la ESP32-CAM
        resultado = random.choice(materiales)
        
        print(f"Simulando detección: Clasificado como {resultado['material']} con {resultado['confidence']}% de confianza.")
        
        # Emite el evento de clasificación a la página web a través de WebSocket
        socketio.emit('clasificacion', {'type': 'clasificacion', 'payload': resultado})

# Se ejecuta al iniciar el servidor
@socketio.on('connect')
def handle_connect():
    print('Cliente conectado al WebSocket. Iniciando la simulación...')
    emit('mensaje', {'payload': {'message': 'Conectado al servidor ecoTrash. Esperando un objeto...'}})

if __name__ == '__main__':
    # Inicia el hilo de simulación de la clasificación
    # El servidor web correrá en el hilo principal
    threading.Thread(target=simular_y_clasificar).start()
    
    # Ejecuta la aplicación de Flask con SocketIO
    # Asegúrate de tener el archivo index.html en la carpeta 'templates'
    # Nota: la ESP32-CAM en tu código real tendrá una IP específica y el video se verá en la URL
    # del atributo 'src' de la imagen en el HTML.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)