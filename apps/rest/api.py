# -*- coding: utf-8 -*-
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import base64
from decimal import *
from flask import Flask, make_response, request, Response
import glob
import io
import json
import os
import werkzeug
from datetime import datetime
import numpy as np

from predict import Predict

api = Flask(__name__)

api.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

UPLOAD_DIR = os.getenv("UPLOAD_DIR_PATH") or 'uploads'

BEST_SCORE_WEIGHTS_FILE = sorted(glob.glob('/work/*.h5'), reverse=True)[0]

pred = Predict(BEST_SCORE_WEIGHTS_FILE)

'''
target_classes = ['burger', 'cheese', 'chicken_curry', 'chocolate', 'corn', 'ebimayo', 'lemon', 'mentai', 'natto',
                  'premium_cheese', 'premium_mentai', 'premium_steak', 'rusk', 'salad', 'salami', 'takoyaki', 'tongue', 'tonkatsu',
                  'yakitori']
'''
target_classes = ['テリヤキバーガー', 'チーズ', 'チキンカレー', 'チョコレート', 'コンポタージュ', 'エビマヨネーズ',
                  'レモンスカッシュ', 'めんたい', 'なっとう', 'プレミアムチーズ', 'プレミアム明太子', 'プレミアムステーキ',
                  'シュガーラスク', 'やさいサラダ', 'サラミ', 'たこ焼き', '牛タン塩', 'とんかつソース', 'やきとり']


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@api.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        make_response(jsonify({'result':'uploadFile is required.'}))

    file = request.files['file']
    filename = file.filename
    if '' == filename:
        make_response(jsonify({'result':'filename must not empty.'}))

    save_filename = datetime.now().strftime('%Y%m%d_%H%M%S_') \
                   + werkzeug.utils.secure_filename(filename)
    save_filepath = os.path.join(UPLOAD_DIR, save_filename)
    file.save(save_filepath)

    scores, cam, guided = pred(save_filepath)
    print(scores)
    y_preds = (-scores).argsort()[:5]   #np.argmax(scores)
    pred_results = [[i, target_classes[i]] for i in y_preds]
    print(pred_results)
    
    gradcam_img = array_to_img(cam)
    img_bytes = io.BytesIO()
    gradcam_img.save(img_bytes, format='PNG')
    guided_img = array_to_img(guided)
    guided_img_bytes = io.BytesIO()
    guided_img.save(guided_img_bytes, format='PNG')

    json_string = json.dumps({
        'scores':scores,
        'predict_top5':pred_results,
        'gradcam_image':base64.b64encode(img_bytes.getvalue()).decode('utf-8'),
        'guided_gradcam_image':base64.b64encode(guided_img_bytes.getvalue()).decode('utf-8')
    }, cls=NumpyEncoder)
    res = make_response(json_string)
    
    res.headers['Content-Type'] = 'text/json'
    res.headers['Access-Control-Allow-Origin'] = 'http://localhost:13000'
    res.status_code = 200
    return res

@api.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    return 'result : file size is overed.'

# main
if __name__ == "__main__":
    print(api.url_map)
    api.run(host='0.0.0.0', port=3001)
