from flask import Flask, render_template
from flask import Response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    def generate():
        # Placeholder: Replace with actual video streaming logic
        while True:
            frame = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\xff\xd8\xff\xe0" + b"\r\n"
            yield frame
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
