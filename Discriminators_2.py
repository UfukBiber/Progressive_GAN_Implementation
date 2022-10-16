import tensorflow as tf 
from Custom_Layers import Weighted_Sum, wasserstain_loss, MinibatchStd, Conv2D, Dense

def Discriminator_Base(inp_shape = (4, 4), leaky_relu_alpha = 0.2):
    ## Input Layer of Discriminator
    inp = tf.keras.layers.Input(shape = inp_shape + (3,))
    out = Conv2D(128,  1)(inp)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)

    ## First Block
    out = Conv2D(128, 3)(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = Conv2D(128, 3)(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)

    ## Last Block Of Discriminator

    out = MinibatchStd()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.zeros())(out)
    base_model = tf.keras.models.Model(inp, out)
    base_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3, 0, 0.999, 1e-8), loss = wasserstain_loss)
    return base_model

def Add_Discriminator_Block(old_model, leaky_relu_alpha = 0.2):
    old_input_shape = list(old_model.input_shape)

    ## New Input Layer   
    inp = tf.keras.layers.Input(shape = (old_input_shape[1] * 2, old_input_shape[2] * 2, 3))

    ## New Block
    
    out = Conv2D(128, 3)(inp)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = Conv2D(128, 3)(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = tf.keras.layers.AveragePooling2D()(out)
    residual = out 
    for i in range(3, len(old_model.layers)):
        out = old_model.layers[i](out)
    ## Standart Model
    standart_model = tf.keras.models.Model(inp, out)
    standart_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3, 0., 0.999, 1e-8), loss = wasserstain_loss)

    ## Fade in Model
    fade_in_input = tf.keras.layers.AveragePooling2D()(inp)
    fade_in_out = old_model.layers[1](fade_in_input)
    fade_in_out = old_model.layers[2](fade_in_out)
    fade_in_out = Weighted_Sum()([fade_in_out, residual])
    for i in range(3, len(old_model.layers)):
        fade_in_out = old_model.layers[i](fade_in_out)
    
    fade_in_model = tf.keras.models.Model(inp, fade_in_out)
    fade_in_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3, 0., 0.999, 1e-8), loss = wasserstain_loss)

    return [standart_model, fade_in_model]


def Get_Discriminator_Models(n_block, inp_shape = (4, 4), leaky_relu_alpha = 0.2):
    models = list()
    base_model = Discriminator_Base(inp_shape, leaky_relu_alpha)
    models.append([base_model, base_model])

    for i in range(1, n_block):
        models.append(Add_Discriminator_Block(models[i-1][0], leaky_relu_alpha))
    return models




if __name__ == "__main__":
    discrimator_models = Get_Discriminator_Models(3)
    for i in range(len(discrimator_models)):
        print(discrimator_models[i][0].input_shape)