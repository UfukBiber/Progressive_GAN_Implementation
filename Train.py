import tensorflow as tf
from Custom_Layers import Weighted_Sum 
from Input import GenerateDataset
from Generators_2 import Get_Generator_Models
from Discriminators_2 import Get_Discriminator_Models
from Gan import Generate_Gan_Models
import matplotlib.pyplot as plt

N_IMAGES= 30000
BATCH_SIZES = {4 : 16, 8:16, 16 : 16, 32 : 8, 64 : 4, 128 : 2, 256:2, 512 : 1}
EPOCHS = {4 : 12, 8 : 12, 16 : 8, 32 : 8, 64 : 6, 128 : 6, 256 : 4, 512 : 4}
N_BLOCKS = 8
LATENT_DIM = 256

class ProgressiveGan():
    def __init__(self, generators, discriminators, gans):
        self.generators = generators
        self.discriminators = discriminators
        self.gans = gans 
    
    def train_model(self, generator, discriminator, gan, ds, is_fade_in = False, n_batches = 32, inp_size = 4):
        i = 0
        total_batches = N_IMAGES / n_batches
        epochs = EPOCHS[inp_size]
        total_steps = total_batches * epochs 
        
        for epoch in range(epochs):
            for batch in ds:
                step = epoch * total_batches + i 
                if is_fade_in and (step == total_steps // 5):
                    self.Update_WeighSum_Alpha(generator, discriminator, step, total_steps)
                batch_size = tf.shape(batch)[0]
                fake_label = -tf.ones(shape = (batch_size, 1), dtype = tf.float32)
                real_label = tf.ones(shape = (batch_size, 1), dtype = tf.float32)
                latent_vector = tf.random.normal(shape = (batch_size, LATENT_DIM))
                gen_img = generator(latent_vector)

                ## Discriminator Training
                fake_disc_loss = discriminator.train_on_batch(gen_img, fake_label)
                real_disc_loss = discriminator.train_on_batch(batch, real_label)

                ## Gan Training
                latent_vector = tf.random.normal(shape = (batch_size, LATENT_DIM))
                gan_loss = gan.train_on_batch(latent_vector, real_label)

                print("Epoch : %i \ %i  Batch : %i \ %i  Gan_Loss : %.3f   Fake_Disc_Loss : %.3f     Real_Disc_Loss : %.3f"%(epoch + 1, epochs, i+1, total_batches, gan_loss, fake_disc_loss, real_disc_loss), end = "\r")
                i+=1
            self.Save_Weights(generator, discriminator, is_fade_in, inp_size)
            print()
            i = 0
            if not is_fade_in:
                self.Generate_Image(generator, inp_size, epoch)

    def Generate_Image(self, generator, inp_size, epoch):
        latent_vector = tf.random.normal((3, LATENT_DIM))
        output = generator(latent_vector)
        output *= 127.5
        output += 127.5
        output = tf.cast(output, tf.uint8)
        output = output.numpy()
        for i in range(len(output)):
            plt.imsave("Generated_Images/Image_%i_%i_%i.jpg"%(inp_size, epoch, i), output[i])


    def Train(self):
        print("Training 4x4 input model")
        ds = GenerateDataset(N_IMAGES, (4, 4), BATCH_SIZES[4])
        self.train_model(self.generators[0][0], self.discriminators[0][0], self.gans[0][0], ds, False, BATCH_SIZES[4], 4)
        for i in range(1, len(self.generators)):
            input_shape = 2 ** (i+2)
            ds = GenerateDataset(N_IMAGES, (input_shape, input_shape), BATCH_SIZES[input_shape])
            print("Training %ix%i fade in model"%(input_shape, input_shape))
            self.train_model(self.generators[i][1], self.discriminators[i][1], self.gans[i][1], ds, True, BATCH_SIZES[input_shape], input_shape)
            print("Training %ix%i standart model"%(input_shape, input_shape))
            self.train_model(self.generators[i][1], self.discriminators[i][1], self.gans[i][1], ds, False, BATCH_SIZES[input_shape], input_shape)
        print("")
    
    def Update_WeighSum_Alpha(self, generator, discriminator, step, n_batches):
        total_steps = n_batches * 3
        for layer in generator.layers:
            if (isinstance(layer, Weighted_Sum)):
                layer.alpha = step / total_steps
        for layer in discriminator.layers:
            if (isinstance(layer, Weighted_Sum)):
                layer.alpha = step / total_steps
    
    def Save_Weights(self, generator, discriminator, is_fade_in, inp_size):
        if (not is_fade_in):
            generator.save_weights("Generators/Generator_%i_Standart"%inp_size)
            discriminator.save_weights("Discriminators/Discriminator_%i_Standart"%inp_size)
        else:
            generator.save_weights("Generators/Generator_%i_Fade_in"%inp_size)
            discriminator.save_weights("Discriminators/Discriminator_%i_Fade_in"%inp_size)
            
        
    

            

if __name__ == "__main__":
    generators = Get_Generator_Models(N_BLOCKS, LATENT_DIM, (4, 4))
    discriminators = Get_Discriminator_Models(N_BLOCKS)
    gans = Generate_Gan_Models(discriminators, generators)
    prog_gan = ProgressiveGan(generators, discriminators, gans)
    prog_gan.Train()


