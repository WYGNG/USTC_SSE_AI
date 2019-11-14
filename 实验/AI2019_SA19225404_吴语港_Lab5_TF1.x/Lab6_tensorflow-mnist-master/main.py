import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model
from tensorflow.lite.python import lite 

x = tf.placeholder("float", [None, 784])
placeholder_y = np.zeros((1,28,28,1)).reshape(1,28,28,1).astype('uint8')
sess = tf.Session()

# restore trained data
with tf.variable_scope("userdefinefunction"):
    y0 = model.userdefinefunction(placeholder_y)
    
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")

def userdefinefunction(input):
    return sess.run(y0, feed_dict={placeholder_y: input}).flatten().tolist()

def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = (np.array(request.json, dtype=np.uint8)).reshape(1, 784)
    output0 = (model.userdefinefunction(input.reshape(1,28,28,1).astype('uint8'))/255.0).tolist()
    output1 = convolutional(((255-input)/255.0).reshape(1, 784))
    return jsonify(results=[output0, output1])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)
