from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

ghgmodel = "./GHG_FineTune/checkpoint/cp1"

tokenizer = T5Tokenizer.from_pretrained(ghgmodel)
model = T5ForConditionalGeneration.from_pretrained(ghgmodel)


@app.route('/', methods=['GET'])
def process_get_request():
    return "hello", 200


@app.route('/', methods=['POST'])
def process_post_request():
    if request.is_json:
        data = request.get_json()
        if 'my_question' in data:
            input_text = data['my_question']
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=200)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({"answer": generated_text}), 200
        else:
            return jsonify({"error": "Missing 'my_question' in JSON data"}), 400
    else:
        return jsonify({"error": "Invalid request, must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True)
