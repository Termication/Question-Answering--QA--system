from flask import Flask, request, render_template
from qa_utils import load_documents, split_documents, create_vector_store, ask_question

app = Flask(__name__)

# Load and prepare data at startup
docs = load_documents()
texts = split_documents(docs)
vstore = create_vector_store(texts)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = ask_question(question, vstore)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
