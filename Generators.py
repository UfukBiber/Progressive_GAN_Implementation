import tensorflow as tf 
from Custom_Layers import PixelNormalization, Conv2D, Fade_In, Dense

IMAGE_DIM_TO_N_STEP = dict((2 ** i, i - 2) for i in range(2, 10))
N_STEP_TO_IMAGE_DIM = dict((i - 2, 2 ** i) for i in range(2, 10))
IMAGE_DIMS = list(IMAGE_DIM_TO_N_STEP.keys())

class Generator:
    def __init__(self, latent_dim, leaky_relu_alpha, n_steps):
        self.latent_dim = latent_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.generator_blocks = []
        self.to_rgbs = []
        self.Build_All_Blocks(n_steps)

    def Generate_Base_Block(self):
        input = tf.keras.layers.Input(shape = (self.latent_dim, ), name = "generator_input")

        output = PixelNormalization()(input)
        output = Dense(4 * 4 * 128, gain = 1./8)(output)
        output = tf.keras.layers.Reshape((4, 4, 128))(output)

        output = Conv2D(128, 4, name = "base_block_conv_1")(output)
        output = PixelNormalization()(output)
        output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(output)

        output = Conv2D(128, 3, name = "base_block_conv_2")(output)
        output = PixelNormalization()(output)
        output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(output)

        base_model = tf.keras.models.Model(input, output, name = "base_block")
        self.generator_blocks.append(base_model)


    def Build_Block(self, img_size):
        inp_shape = (int(img_size/2), int(img_size/2), 128)
        new_input = tf.keras.layers.Input(shape = inp_shape)
        
        ## New Block 
        new_output = tf.keras.layers.UpSampling2D()(new_input)
        new_output = Conv2D(128, 3, name = "%i_%i_conv1"%(img_size, img_size))(new_output)
        new_output = PixelNormalization()(new_output)
        new_output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(new_output)

        new_output = Conv2D(128, 3, name = "%i_%i_conv2"%(img_size, img_size))(new_output)
        new_output = PixelNormalization()(new_output)
        new_output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(new_output)

        new_block = tf.keras.models.Model(new_input, new_output, name = "%i_%i_block"%(img_size, img_size))
        self.generator_blocks.append(new_block)
    
    def Build_All_Blocks(self, n_steps):
        self.Generate_Base_Block()
        for i in range(1, n_steps):
            img_size = N_STEP_TO_IMAGE_DIM[i]
            self.Build_Block(img_size)
        

    def Generate_To_RGB(self, filter_num, n_step):
        img_dim = N_STEP_TO_IMAGE_DIM[n_step]
        inp = tf.keras.layers.Input(shape = (img_dim, img_dim, filter_num))
        out = Conv2D(3, 1, gain = 1., name = "%i_%i_To_RGB_Conv"%(img_dim, img_dim))(inp)
        return tf.keras.models.Model(inp, out, name = "%i_%i_To_RGB"%(img_dim, img_dim))
    
    def Build_Standart_Model(self, n_step):
        img_dim = N_STEP_TO_IMAGE_DIM[n_step]
        inp = tf.keras.layers.Input(shape = (self.latent_dim))
        output = self.generator_blocks[0](inp)
        for i in range(1, n_step+1):
            output = self.generator_blocks[i](output)
        to_rgb = self.Generate_To_RGB(128, n_step)
        self.to_rgbs.append(to_rgb)
        output = to_rgb(output)
        return tf.keras.models.Model(inp, output, name = "%i_%i_Standart_Gen_Model"%(img_dim, img_dim))
    
    def Build_Fade_In_Model(self, n_step):
        img_shape = N_STEP_TO_IMAGE_DIM[n_step]
        inp = tf.keras.layers.Input(shape = (self.latent_dim, ), name = "generator_inp")
        alpha_input = tf.keras.layers.Input(shape = (1), name = "alpha_input")
        old_to_rgb = self.to_rgbs[n_step - 1]
        new_to_rgb = self.to_rgbs[n_step]
        output = self.generator_blocks[0](inp)
        for i in range(1, n_step):
            output = self.generator_blocks[i](output)
        old_output = old_to_rgb(output)
        old_output = tf.keras.layers.UpSampling2D()(old_output)
        new_output = self.generator_blocks[n_step](output)
        new_output = new_to_rgb(new_output)
        output = Fade_In()(old_output, new_output, alpha_input)
        return tf.keras.models.Model((inp, alpha_input), output, name = "%i_%i_Fade_In__Gen_Model"%(img_shape, img_shape))

    def Save_Model(self, model:tf.keras.models.Model):
        model.save_weights("Generator/%s"%(model.name))
    
    def Load_Model(self, img_dim):
        n_step = IMAGE_DIM_TO_N_STEP[img_dim]
        model = self.Build_Standart_Model(n_step)
        model.load_weights("Generator/%s"%(model.name))
        return model


if __name__ == "__main__":
    print(IMAGE_DIM_TO_N_STEP)
    print(IMAGE_DIMS)
    generator = Generator(128, 0.2, 5)
    for i in range(2):
        standart_model = generator.Build_Standart_Model(i)
        standart_model.summary()
        print("\n")
    fade_in_model = generator.Build_Fade_In_Model(1)
    fade_in_model.summary()



    
        