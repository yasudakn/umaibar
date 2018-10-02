from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.backend as K
from tensorflow.python.framework import ops
import keras
import tensorflow as tf

import cv2
import numpy as np

class Predict:
    bs = 32
    nb_classes=19
    im_rows, im_cols = (224, 224)
    model = None
    TEST_DIR = "/work/umaibar/data-content/test"

    def __init__(self, weights):
        # モデルの構築とImageNetで学習済みの重みの読み込み
        base_model = VGG16(include_top=False, weights='imagenet', pooling=None)

        # FC層を構築
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.nb_classes, activation='softmax')(x)
        self.model =  Model(inputs=base_model.input, outputs=predictions)

        # 上で出来上がったval_lossが一番小さいモデルの重みファイルの読み込み
        self.model.load_weights(weights, by_name=False)
        
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Guided Grad-CAM
        register_gradient()
        guided_model = modify_backprop(self.model, 'GuidedBackProp')

        guided_model.load_weights(weights, by_name=False)

        guided_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.saliency_fn = compile_saliency_function(guided_model, activation_layer='block5_conv3')

        # test-data iterator
        test_batchs = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            self.TEST_DIR,
            target_size = (self.im_rows, self.im_cols),
            batch_size = self.bs,
            color_mode = 'rgb',
            shuffle = False,
            class_mode = "categorical"
        )

        x_test = np.array([])
        x_test.resize((0, self.im_rows, self.im_cols, 3))
        for i in range(int(test_batchs.samples/self.bs) + 1):
            imgs, labels = next(test_batchs)
            x_test = np.vstack((x_test, imgs))

        self.test_datagen = ImageDataGenerator(
            rescale = 1.0 / 255,
            featurewise_center = True,
            featurewise_std_normalization = True,
            zca_whitening = False
        )
        self.test_datagen.fit(x_test)

    def __call__(self, path, gradcam_layer = "block5_conv3"):
        img = img_to_array(load_img(path, target_size=(self.im_rows, self.im_cols)))
        img = np.expand_dims(img, axis=0)
        img_data = self.test_datagen.flow(img, batch_size=1, shuffle=False)
        x = next(img_data)
        cam, heatmap, pred_scores = grad_cam(self.model, img, x, gradcam_layer, *(self.im_rows, self.im_cols))
        saliency = self.saliency_process(x)
        guided = saliency[0] * heatmap[..., np.newaxis]

        return pred_scores[0], cam[0], guided[0]
    
    def saliency_process(self, x):
        return self.saliency_fn([x])

def grad_cam(model, image, x, layer_name, *sizes):
    im_rows, im_cols = (sizes[0], sizes[1])
    # predict target image
    predictions = model.predict(x, batch_size=1, verbose=0)
    class_idx = np.argmax(predictions[0])
    loss = model.output[:, class_idx]

    conv_output = model.get_layer(layer_name).output
    
    grads = K.gradients(loss, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])

    # get gradients
    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # mean and dot
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(output, weights)

    # create heatmap
    cam = cv2.resize(cam, (im_rows, im_cols), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # coloring
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # to RGB
    cam = (np.float32(cam) + image) # composite original image
    
    return cam, heatmap, predictions
  
def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name, nb_classes=19):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        base_model = VGG16(include_top=False, weights='imagenet', pooling=None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        new_model =  Model(inputs=base_model.input, outputs=predictions)
        
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
