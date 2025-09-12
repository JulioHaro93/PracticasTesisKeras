#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocesar_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def grad_cam(model, img_array, class_index, conv_layer_name="conv5_block3_out"):
    base_model = model.layers[0]  # extraer ResNet50
    conv_layer = base_model.get_layer(conv_layer_name)

    # modelo combinado
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)
    return superimposed_img

def grad_cam_main(img_path, model_path):

    model = tf.keras.models.load_model(model_path)
    model.summary()
    if not model.built:
        model.build(input_shape=(None, 224, 224, 3))
    try:
        _ = model(np.zeros((1, 224, 224, 3), dtype=np.float32))
    except Exception as e:
        print("Modelo ya estaba construido:", e)
    img_array = preprocesar_img(img_path)
    preds = model.predict(img_array)

    class_index = int(preds[0] > 0.5)
    heatmap = grad_cam(model, img_array, class_index)
    superimposed_img = overlay_heatmap(img_path, heatmap)

    # Mostrar resultado
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicción clase {class_index}")
    plt.axis('off')
    plt.show()
