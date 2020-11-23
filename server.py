from instance_segmentation import shape_estimation
from flask import Flask
from flask import request
from flask import Response

app = Flask(__name__)


@app.route("/")
def hello():
    return Response(shape_estimation(request.args["video_path"]), mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
