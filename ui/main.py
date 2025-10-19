from flask import Flask, render_template, request
from ai.predict import predict_comment  

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    comment = ""

    if request.method == "POST":
        comment = request.form.get("comment", "")
        if comment.strip():
            result = predict_comment(comment)
        else:
            result = "Введите текст комментария ..."

    return render_template("index.html", result=result, comment=comment)

if __name__ == "__main__":
    app.run(debug=True)