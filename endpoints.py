from flask import request, send_file
from dynaconf import settings
from demo_stgan_api import facial_attribute_editing
from flask import Blueprint
from flask import jsonify
import os
import uuid

demo = Blueprint('demo', __name__)


@demo.route('/hello', methods=['GET', 'POST'])
def hello():
    response = {'errcode': 0, 'errmsg': 'Hello World'}
    return jsonify(response), 200


@demo.route('/face', methods=['POST'])
def facial_attribute_editing_api():
    files = request.files
    if files.get("img") is None:
        response = {'errcode': -1, 'errmsg': 'invalid param'}
    else:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        suffix = files['img'].filename.rsplit('.', 1)[1]
        prefix = str(uuid.uuid4())
        server_img_name = '{}.{}'.format(prefix, suffix)
        server_img_path = os.path.join(img_path, server_img_name)
        with open(server_img_path, 'wb') as wbf:
            wbf.write(files['img'].read())
        server_img_output_name = '{}{}.{}'.format(prefix, '-edit', suffix)
        server_img_output_path = os.path.join(img_path, server_img_output_name)
        result = facial_attribute_editing.edit(server_img_path, server_img_output_path)
        if result:
            mimetype = 'image/{}'.format(suffix)
            return send_file(server_img_output_path, mimetype=mimetype)
        else:
            response = {'errcode': -1, 'errmsg': 'internal error'}
    return jsonify(response), 200
