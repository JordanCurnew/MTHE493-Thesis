import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from scipy.stats import norm


class CustomCompression(Model):
    def __init__(self , alpha, analysis, synthesis):
        super(CustomCompression, self).__init__()
        self.analysis = analysis # encoder layers
        self.synthesis = synthesis # decoder layers

        self.alpha = alpha # distortion loss weight

        self.var = [] 
        self.mu = []

    def call(self, inputs, training=True):
        x = self.analysis(inputs)
        if training:
          x = self.add_noise(x)
        else:
          x = self.quantize(x)
        return self.synthesis(x)

    def gaussian(self, latent):
        result = 1/2*tf.experimental.numpy.log2(2 * 3.1415926535 * self.var) + tf.math.pow((latent-tf.cast(self.mu, dtype=tf.float32)), 2)/(2 * self.var)*np.log2(2.712)
        return result

    def add_noise(self, latent):
        return latent + tf.random.uniform(tf.shape(latent), -1/2, 1/2)

    def quantize(self, latent):
        return tf.math.floor(latent + 1/2)
    
    def actual_entropies(self, quantized_latent):
        entropies = []
        for i in range(quantized_latent.shape[1]):
            unique_values, value_indices, value_counts = tf.unique_with_counts(quantized_latent[:, i])
            entropy = 0

            for count in value_counts.numpy():
                probability = count / quantized_latent.shape[0]
                entropy += -(probability) * np.log2(probability)
                entropies.append(entropy)

        return entropies


    def compression_loss(self, input, output):
        distortion = tf.reduce_mean(tf.square(input - output)) # Consider this to tf.reduce_sum
        latent = self.analysis(input)
        rate = tf.reduce_mean(self.gaussian(self.add_noise(latent))) # Consider this to tf.reduce_sum ## SPENCER CHANGE: was not adding noise to the latent ## good point, mike
        return rate, distortion
    
    def msssim_compression_loss(self, input, output):
        distortion = tf.reduce_mean(tf.image.ssim(input, output, 255, filter_size=11)) # Consider this to tf.reduce_sum
        latent = self.analysis(input)
        rate = tf.reduce_mean(self.gaussian(self.add_noise(latent))) # Consider this to tf.reduce_sum ## SPENCER CHANGE: was not adding noise to the latent ## good point, mike
        return rate, distortion

    def test_loss(self, inputs):
        latent = self.analysis(inputs)
        distortion = tf.reduce_mean(tf.image.ssim(inputs, self(inputs, training=False), 255, filter_size=11))
        rate = tf.reduce_mean(self.gaussian(self.quantize(latent)))
        entropies = self.actual_entropies(self.quantize(latent))
        return rate, distortion, tf.reduce_mean(entropies)

    def msssim_train_step(self, x):
        with tf.GradientTape() as tape:
            y = self(x, training=True)
            rate, distortion = self.msssim_compression_loss(x, y)
            loss = rate - self.alpha * distortion

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, rate, distortion
        
        
    def train_step(self, x):
        with tf.GradientTape() as tape:
            y = self(x, training=True)
            rate, distortion = self.compression_loss(x, y)
            loss = rate + self.alpha * distortion

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, rate, distortion
        # return {"loss": loss}

    def train(self, x_train, validation_data, num_epochs, batch_size=32, learning_rate=0.001):
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

        history = {
            'loss' : [],
            'rate' : [],
            'distortion' : []
        } 

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(num_epochs):
            latent = self.analysis(x_train)
            self.mu = tf.math.reduce_mean(latent, 0)
            self.var = tf.math.reduce_variance(latent, 0)

            epoch_loss = 0.0
            epoch_rate = 0.0
            epoch_distortion = 0.0

            num_batches = 0

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                loss, rate, distortion = self.msssim_train_step(x_batch)
#                 loss, rate, distortion = self.train_step(x_batch)

                epoch_loss += loss
                epoch_rate += rate
                epoch_distortion += distortion

                num_batches += 1
            if epoch % 2 == 0:
                result = self(x_train[:2], training=False)
#                 plot(x_train[0], 1)
#                 plot(result[0], 1)
                result = self(validation_data[:2], training=False)
#                 plot(validation_data[0], 1)
#                 plot(result[0], 1)
            
            epoch_test_rate, epoch_test_distortion, epoch_test_entropy = self.test_loss(validation_data)

            epoch_loss /= num_batches
            epoch_rate /= num_batches
            epoch_distortion /= num_batches
            print("Epoch {}: Loss = {}, Rate = {}, Distortion = {}".format(epoch+1, epoch_loss, epoch_rate, epoch_distortion))
            print("          Test BPP = {}, Test Rate = {}, Test Distortion = {}".format(epoch_test_entropy, epoch_test_rate, epoch_test_distortion))
            history["loss"].append(np.mean(epoch_loss))
            history["rate"].append(np.mean(epoch_rate))
            history["distortion"].append(np.mean(epoch_distortion))
        return history
