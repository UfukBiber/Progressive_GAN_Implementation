import tensorflow as tf 
from Custom_Layers import PixelNormalization, Conv2D, Weighted_Sum, Dense

def Generator_Base(latent_dims, inp_dims, leaky_relu_alpha = 0.2):

    ## Base Of Generator

    inp = tf.keras.layers.Input(shape = (latent_dims, ))
    out = Dense(latent_dims * inp_dims[0] * inp_dims[0])(inp)
    out = tf.keras.layers.Reshape((inp_dims[0], inp_dims[0], latent_dims))(out)
    out = Conv2D(128, 4, gain = 2)(out)
    out = PixelNormalization()(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = Conv2D(128, 3, gain = 2)(out)
    out = PixelNormalization()(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)

    ## Last layer of generator

    out = Conv2D(3, 1)(out)

    base_model = tf.keras.models.Model(inp, out)
    return base_model 

def Add_Generator_Block(old_model:tf.keras.models.Model, leaky_relu_alpha = 0.2):
    ## Input of new block
    inp = old_model.layers[-2].output

    ## New Block
    scaled_inp = tf.keras.layers.UpSampling2D()(inp)
    out = Conv2D(128, 3)(scaled_inp)
    out = PixelNormalization()(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = Conv2D(128, 3)(out)
    out = PixelNormalization()(out)
    out = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(out)
    out = Conv2D(3, 1)(out)
    residual = out 
    ## Standart Model
    standart_model = tf.keras.models.Model(old_model.input, out)
    ## Fade-In Model
    old_model_output = old_model.output 
    scaled_old_model_output = tf.keras.layers.UpSampling2D()(old_model_output)
    combined_output = Weighted_Sum()([scaled_old_model_output, residual])
    fade_in_model = tf.keras.models.Model(old_model.input, combined_output)
    return [standart_model, fade_in_model]

def Get_Generator_Models(n_block, latent_dims = 128, inp_dims = (4, 4), leaky_relu_alpha = 0.2):
    models = list()
    base_model = Generator_Base(latent_dims, inp_dims, leaky_relu_alpha)
    models.append([base_model, base_model])
    for i in range(1, n_block):
        models.append(Add_Generator_Block(models[i-1][0], leaky_relu_alpha))
    return models



if __name__ == "__main__":
    models = Get_Generator_Models(3)
    inp = tf.random.normal(shape = (32, 128))
    # for i in range(len(models)):
    #     # output = models[i][0](inp)
    #     # print(tf.shape(output))
    models[1][0].summary()
        




