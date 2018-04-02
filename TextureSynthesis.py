import numpy as np
import tensorflow as tf
import sys

from ImageUtils import *
from model import *

SAVE_STEP = 1000

class TextureSynthesis:
    def __init__(self, sess, model, actual_image, layer_constraints, model_name, image_name, saveDir, iterations, nSplits):
        # 'layer_constraints' is dictionary with key = VGG layer and value = weight (w_l)
        # 'sess' is tensorflow session
        self.model_name = model_name # Of the form: conv#
        self.image_name = image_name # Of the form: imageName

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

        self.model = model # Model instance
        assert self.model.model_initialized(), "Model not created yet."
        self.model_layers = self.model.get_model()

        # Layer weights for the loss function
        self.layer_weights = layer_constraints

        self.actual_image = actual_image # 256x256x3

        self.init_image = self._gen_noise_image()

        # Number of splits
        self.nSpl = nSplits

        self.constraints = self._get_constraints() # {layer_name: activations}

        # Directory to save outputs in
        self.saveDir = saveDir

        # Number of iterations to run for
        self.iterations = iterations

    def get_texture_loss(self):
        total_loss = 0.0
        for layer in self.layer_weights.keys():
            print layer
            layer_activations = self.model_layers[layer]
            layer_activations_shape = layer_activations.get_shape().as_list()
            assert len(layer_activations_shape) == 4 # (1, H, W, outputs)
            assert layer_activations_shape[0] == 1, "Only supports 1 image at a time."
            num_filters = layer_activations_shape[3] # N
            num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M
            layer_gram_matrix = self._compute_gram_matrix_subset(layer_activations, num_filters, num_spatial_locations, self.nSpl)
            desired_gram_matrix = self.constraints[layer]

            total_loss += self.layer_weights[layer] * (1.0 / (4 * (num_filters**2) * (num_spatial_locations**2))) \
                          * tf.reduce_sum(tf.pow(desired_gram_matrix - layer_gram_matrix, 2))
        return total_loss

    def _get_constraints(self):
        self.sess.run(tf.initialize_all_variables())
        constraints = dict()
        for layer in self.layer_weights:
            self.sess.run(self.model_layers['input'].assign(self.actual_image))
            layer_activations = self.sess.run(self.model_layers[layer])
            num_filters = layer_activations.shape[3] # N
            num_spatial_locations = layer_activations.shape[1] * layer_activations.shape[2] # M
            print layer_activations.shape
            constraints[layer] = self._compute_gram_matrix_subset_np(layer_activations, num_filters, num_spatial_locations, self.nSpl)

        return constraints

    def _compute_gram_matrix_np(self, F, N, M):
        F = F.reshape(M, N)
        return np.dot(F.T, F)

    def _compute_gram_matrix(self, F, N, M):
        # F: (1, height, width, num_filters), layer activations
        # N: num_filters
        # M: number of spatial locations in filter (filter size ** 2)
        F = tf.reshape(F, (M, N)) # Vectorize each filter so F is now of shape: (height*width, num_filters)
        return tf.matmul(tf.transpose(F), F) 

    def _compute_gram_matrix_subset_np(self, F, N, M, n_spl):
        h = F.shape[1]
        w = F.shape[2]

        F = np.squeeze(F);
        F2 = np.zeros((N,N,1))

        sub_sz = h/n_spl;

        for hi in range(0, h, sub_sz):
            end_h = np.minimum(hi + sub_sz, h)

            for wi in range(0, w, sub_sz):
                end_w = np.minimum(wi + sub_sz, w)

                subset = F[hi:end_h, wi:end_w, :] # s x s x n_filt
                subset = subset.reshape((len(range(hi, end_h)) * len(range(wi, end_w)), N)) # s^2 x n_filt

                # Take the dot product within that subregion.
                dp = np.dot(subset.T, subset).reshape((N,N,1)) # size: n_filt x n_filt
                F2 = np.concatenate((F2, dp), axis=2)

        return F2[:,:,1:];

    def _compute_gram_matrix_subset(self, F, N, M, n_spl):
        # F: (1, height, width, num_filters) -- layer activations
        # N: num_filters
        # M: number of spatial locations in filter (height * width)
        # n_subsets: number of subsets of each filter.

        # Multiply a (nFilters, filtSz) x (filtSz, nFilters) to get a (nFilt x nFilt) gram matrix 
        # where each element (i,j) represents the dot product of filter_i with filter_j

        # I want to change this so you're only computing the dot product within a subset of the image.
        f_shape = shape(F)
        h = f_shape[1]
        w = f_shape[2]

        F = tf.squeeze(F);

        F2 = tf.to_float(tf.constant(np.zeros(N*N), shape=(N, N, 1)));

        sub_sz = h/n_spl;

        ### MAKE CHANGES HERE: AKSHAY
        for hi in range(0, h, sub_sz):
            end_h = np.minimum(hi + sub_sz, h)

            for wi in range(0, w, sub_sz):
                end_w = np.minimum(wi + sub_sz, w)

                subset = F[hi:end_h, wi:end_w, :] # s x s x n_filt
                subset = tf.reshape(subset, (len(range(hi, end_h)) * len(range(wi, end_w)), N)) # s^2 x n_filt

                # Take the dot product within that subregion.
                dp = tf.matmul(tf.transpose(subset), subset)
                dp = tf.reshape(dp, (N,N,1)) # size: n_filt x n_filt
                F2 = tf.concat(values=[F2, dp], axis=2)
            
        return F2[:,:,1:];


    def _gen_noise_image(self):
        input_size = self.model_layers["input"].get_shape().as_list()
        return np.random.randn(input_size[0], input_size[1], input_size[2], input_size[3])

    def train(self):
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.actual_image))

        content_loss = self.get_texture_loss()
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(content_loss)

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.init_image))
        for i in range(self.iterations):
            self.sess.run(train_step)
            if i % 1 == 0:
                print "Iteration: " + str(i) + "; Loss: " + str(self.sess.run(content_loss))
            if i % SAVE_STEP == 0:
                print "Saving image..."
                curr_img = self.sess.run(self.model_layers["input"])
                filename = self.saveDir + "/%dx%d_%s_%s_step_%d" % (self.nSpl, self.nSpl, self.model_name, self.image_name, i)
                save_image(filename, curr_img)
            sys.stdout.flush()

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

#===============================
if __name__ == "__main__":
    sess = tf.Session()
    pool1_weights = {'conv1_1': 1e9, 'pool1': 1e9, 'pool2': 1e9, 'pool3': 1e9, 'pool4': 1e9}
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()
    texture = "tulips.npy"
    image_name = texture.split(".")[0]
    textures_directory = "orig_ims"
    filename = textures_directory + "/" + texture
    img = np.load(filename)
    model_name = 'pool4'
    saveDir = 'v4'
    iterations = 10000
    nSplits = 2
    ts = TextureSynthesis(sess, my_model, img, pool1_weights, model_name, image_name, saveDir, iterations, nSplits)

