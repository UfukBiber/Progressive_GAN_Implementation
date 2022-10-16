import tensorflow as tf 
from Custom_Layers import Fade_In, Conv2D, Dense, MiniBatch_StDev

IMAGE_DIM_TO_N_STEP = dict((2 ** i, i - 2) for i in range(2, 10))
N_STEP_TO_IMAGE_DIM = dict((i - 2, 2 ** i) for i in range(2, 10))
IMAGE_DIMS = list(IMAGE_DIM_TO_N_STEP.keys())

class Discriminator:
    def __init__(self, leaky_relu_alpha, n_steps):
        self.leaky_relu_alpha = leaky_relu_alpha
        self.discriminator_blocks = [] 
        self.from_rgbs = []
        self.Build_All_Blocks(n_steps)

    def Generate_Base_Block(self):
        input = tf.keras.layers.Input(shape = (4, 4, 128), name = "discriminator_input")

        output = MiniBatch_StDev()(input)

        output = Conv2D(128, 3, name = "base_block_conv_1")(output)
        output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(output)

        output = Conv2D(128, 4, name = "base_block_conv_2")(output)
        output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(output)

        output = tf.keras.layers.Flatten()(output)
        output = Dense(1, gain = 1./8, name = "base_block_dense")(output)

        base_model = tf.keras.models.Model(input, output, name = "base_block")
        self.discriminator_blocks.append(base_model)


    def Build_Block(self, img_size):
        inp_shape = (img_size, img_size, 128)
        new_input = tf.keras.layers.Input(shape = inp_shape)
        
        ## New Block 
        new_output = Conv2D(128, 3, name = "%i_%i_conv1"%(img_size, img_size))(new_input)
        new_output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(new_output)

        new_output = Conv2D(128, 3, name = "%i_%i_conv2"%(img_size, img_size))(new_output)
        new_output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(new_output)

        new_output = tf.keras.layers.AveragePooling2D()(new_output)

        new_block = tf.keras.models.Model(new_input, new_output, name = "%i_%i_block"%(img_size, img_size))
        self.discriminator_blocks.append(new_block)
    
    def Build_All_Blocks(self, n_steps):
        self.Generate_Base_Block()
        for i in range(1, n_steps):
            img_size = N_STEP_TO_IMAGE_DIM[i]
            self.Build_Block(img_size)
        

    def Generate_From_RGB(self,  n_step):
        img_dim = N_STEP_TO_IMAGE_DIM[n_step]
        input = tf.keras.layers.Input(shape = (img_dim, img_dim, 3))
        output = Conv2D(128, 1, gain = 1., name = "%i_%i_From_RGB_Conv"%(img_dim, img_dim))(input)
        output = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(output)
        return tf.keras.models.Model(input, output, name = "%i_%i_From_RGB"%(img_dim, img_dim))
    
    def Build_Standart_Model(self, n_step):
        img_dim = N_STEP_TO_IMAGE_DIM[n_step]
        inp = tf.keras.layers.Input(shape = (img_dim, img_dim, 3), name = "%i_%i_inp"%(img_dim, img_dim))
        from_rgb = self.Generate_From_RGB(n_step)
        self.from_rgbs.append(from_rgb)
        output = from_rgb(inp)
        for i in range(n_step, 0, -1):
            output = self.discriminator_blocks[i](output)
        
        output = self.discriminator_blocks[0](output)
        return tf.keras.models.Model(inp, output, name = "%i_%i_Standart_Disc_Model"%(img_dim, img_dim))
    
    def Build_Fade_In_Model(self, n_step):
        img_shape = N_STEP_TO_IMAGE_DIM[n_step]
        inp = tf.keras.layers.Input(shape = (img_shape, img_shape, 3), name = "%i_%i_inp")
        alpha_input = tf.keras.layers.Input(shape = (1), name = "alpha_input")
        
        output = self.generator_blocks[0](inp)
        for i in range(1, n_step):
            output = self.generator_blocks[i](output)
        old_output = old_to_rgb(output)
        old_output = tf.keras.layers.UpSampling2D()(old_output)
        new_output = self.generator_blocks[n_step](output)
        new_output = new_to_rgb(new_output)
        output = Fade_In()(old_output, new_output, alpha_input)
        return tf.keras.models.Model((inp, alpha_input), output, name = "%i_%i_Fade_In_Disc_Model"%(img_shape, img_shape))

    def Save_Model(self, model:tf.keras.models.Model):
        model.save_weights("Generator/%s"%(model.name))
    
    def Load_Model(self, img_dim):
        n_step = IMAGE_DIM_TO_N_STEP[img_dim]
        model = self.Build_Standart_Model(n_step)
        model.load_weights("Generator/%s"%(model.name))
        return model


if __name__ == "__main__":
    discriminator = Discriminator(0.2, 5)
    for i in range(5):
        model = discriminator.Build_Standart_Model(i)
        model.summary()
        print("\n"*2)