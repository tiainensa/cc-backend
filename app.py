from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('data', methods=['POST'])
def get_data():
    data = request.json
    print(data["name"])
    return {"message": "Data received", "data": data}, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)