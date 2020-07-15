from flask import Flask, render_template, request, jsonify
import main
import requests

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat",methods=['POST'])
def chat():

    user_message = request.form["text"]
    print(user_message)
    response = requests.get("http://localhost:5000/parse", params={"q": user_message})
    response_text = main.chatbot(user_message)
    print(response_text)
    return jsonify({"status":"success","response":response_text})
    #return 'OK'


if __name__ == "__main__":
    app.debug = True
    app.run()