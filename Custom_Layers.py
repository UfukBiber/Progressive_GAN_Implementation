import tensorflow as tf 


class MinibatchStd(tf.keras.layers.Layer): 
    def __init__(self, group_size=4, epsilon=1e-8): 

        super(MinibatchStd, self).__init__() 
        self.epsilon = epsilon 
        self.group_size = group_size 
        
    def call(self, input_tensor): 

        n, h, w, c = input_tensor.shape 
        x = tf.reshape(input_tensor, [self.group_size, -1, h, w, c]) 
        group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False) 
        group_std = tf.sqrt(group_var + self.epsilon) 
        avg_std = tf.reduce_mean(group_std, axis=[1,2,3], keepdims=True) 
        x = tf.tile(avg_std, [self.group_size, h, w, 1]) 

        return tf.concat([input_tensor, x], axis=-1) 


class Weighted_Sum(tf.keras.layers.Layer):
    def __init__(self, alpha = 0, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.alpha = alpha
    
    def call(self, inputs):
        assert len(inputs) == 2, "Input length must be two for Weighted Sum layer."
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]



class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, inputs): 
        return inputs / tf.math.sqrt(tf.reduce_mean(inputs**2, axis=-1, keepdims=True) + 1e-8)


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel=3, gain=2, *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel!=1
        
    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)        
        self.w = self.add_weight(shape=[self.kernel,
                                        self.kernel,
                                        self.in_channels,
                                        self.out_channels],
                                initializer=initializer,
                                trainable=True, name='kernel')
        
        self.b = self.add_weight(shape=(self.out_channels,),
                                initializer='zeros',
                                trainable=True, name='bias')
        
        fan_in = self.kernel*self.kernel*self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)
        
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, self.scale*self.w, strides=1, padding="SAME") + self.b
        return output

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, gain = 2., trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.units = units 
        self.gain = gain
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)        
        self.w = self.add_weight(shape=[self.in_channels,
                                        self.units],
                                initializer=initializer,
                                trainable=True, name='kernel')
        
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True, name='bias')
        
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)
    
    #@tf.function
    def call(self, inputs):
        output = tf.matmul(inputs, self.scale*self.w) + self.b
        return output


def wasserstain_loss(y_true, y_pred):
    return - tf.math.reduce_mean(y_true * y_pred)


class Fade_In(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, old_output, new_output, alpha = 0.0):
        return alpha * new_output + (1 - alpha) * old_output  
    
if __name__ == "__main__":
    # #MiniBatch Test
    # inp_1 = tf.fill((4, 4, 1), 5)
    # inp_2 = tf.fill((4, 4, 1), 7)
    # inp_3 = tf.fill((4, 4, 1), 13)
    # inp_1 = tf.cast(inp_1, tf.float32)
    # inp_2 = tf.cast(inp_2, tf.float32)
    # inp_3 = tf.cast(inp_3, tf.float32)
    # inputs = tf.stack((inp_1, inp_2, inp_3), axis = 0)
    # print(inputs)
    # output = MiniBatch_StDev()(inputs)
    # print(output)

    # ## Weighted Sum Test

    # output = Weighted_Sum(0.6)((inp_1, inp_3))
    # print(output)

    ## PixelNorm Test
    # inputs = tf.random.normal(shape = (3, 4, 4, 5), mean = 0., stddev=1., dtype = tf.float32)
    # output = PixelNormalization()(inputs)
    # print(output)
    

    ## CONV2D test

    # inp = tf.random.normal(shape = (12, 4, 4, 3))
    # output = Conv2D(64, 3)(inp)
    # print(tf.shape(output))

    pass 