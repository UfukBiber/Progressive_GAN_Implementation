from selectors import EpollSelector
from Custom_Layers import Fade_In, wasserstain_loss, PixelNormalization, MinibatchStd, Dense, Conv2D
import tensorflow as tf
from Input import GenerateDataset
import matplotlib.pyplot as plt

IMAGE_DIM_TO_N_STEP = dict((2 ** i, i - 2) for i in range(2, 9))
N_STEP_TO_IMAGE_DIM = dict((i - 2, 2 ** i) for i in range(2, 9))
IMAGE_DIMS = list(IMAGE_DIM_TO_N_STEP.keys())

BATCH_SIZES = {4 : 32, 8 : 32, 16 : 16, 32 : 16, 64 : 8, 128 : 4, 256 : 2}
EPOCHS = {4 : 8, 8 : 8, 16 : 8, 32 : 8, 64 : 8, 128 : 8, 256 : 8}
LATENT_DIM = 256
LEAKY_RELU_ALPHA = 0.2
N_IMAGES = 10000
N_STEPS = 6

class Progressive_Gan(tf.keras.models.Model):
    def __init__(self, latent_dim, leaky_relu_alpha, n_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.alpha = tf.Variable(0.0, trainable = False)
        self.step_per_batch = 0.0
        self.n_steps = n_steps
        self.is_in_transition = False

        self.generator_blocks = list()
        self.discriminator_blocks = list()
        self.to_rgb_blocks = list()
        self.from_rgb_blocks = list()

        self.Build_All_Generator_And_To_RGB_Blocks(n_steps+1)
        self.Build_All_Discriminator_And_From_RGB_Blocks(n_steps+1)

    def Generator_Base(self):
        inp = tf.keras.layers.Input(shape = (self.latent_dim), name = "Generator_Input")

        out = PixelNormalization()(inp)
        out = Dense(4 * 4 * 128)(out)
        out = tf.keras.layers.Reshape((4, 4, 128))(out)
        out = PixelNormalization()(out)
        
        out = Conv2D(128, 4, name = "4-4_Conv_1")(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        out = PixelNormalization()(out)

        out = Conv2D(128, 3, name = "4-4_Conv_2")(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        out = PixelNormalization()(out)

        return tf.keras.models.Model(inp, out, name = "Base_Block")

    def Build_To_RGB(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 128))
        out = Conv2D(3, 1, gain = 1.0)(inp)
        to_rgb = tf.keras.models.Model(inp, out, name = "To_RGB_%ix%i"%(img_res, img_res))
        return to_rgb

    def Build_Generator_Block(self, step):
        assert step >= 1, "First Generator Block must be the base."
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp_shape = (int(img_res / 2), int(img_res / 2), 128)
        inp = tf.keras.layers.Input(shape = inp_shape)
    
        out = tf.keras.layers.UpSampling2D()(inp)

        out = Conv2D(128, 3, name = "Conv_1_%i_%i"%(img_res, img_res))(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        out = PixelNormalization()(out)

        out = Conv2D(128, 3, name = "Conv_2_%i_%i"%(img_res, img_res))(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        out = PixelNormalization()(out)

        return tf.keras.models.Model(inp, out, name = "%i_%i_Gen_Block"%(img_res, img_res))

    def Build_All_Generator_And_To_RGB_Blocks(self, n_steps):
        base_model = self.Generator_Base()
        to_rgb = self.Build_To_RGB(0)
        self.generator_blocks.append(base_model)
        self.to_rgb_blocks.append(to_rgb)
        for i in range(1, n_steps):
            block = self.Build_Generator_Block(i)
            to_rgb = self.Build_To_RGB(i)
            self.generator_blocks.append(block)
            self.to_rgb_blocks.append(to_rgb)
    
    def Build_Standart_Generator(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (self.latent_dim, 1), name = "Generator_Input")
        alpha = tf.keras.layers.Input(shape = (1), name = "alpha")
        base_model = self.generator_blocks[0]
        to_rgb = self.to_rgb_blocks[step]
        out = base_model(inp)
        for i in range(1, step+1):
            out = self.generator_blocks[i](out)
        out = to_rgb(out)
        return tf.keras.models.Model([inp, alpha], out, name = "%ix%i_Standart_Generator"%(img_res, img_res))

    def Build_Fade_In_Generator(self, step):
        assert step >= 1
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (self.latent_dim, 1), name = "Generator_Input")
        alpha = tf.keras.layers.Input(shape = (1), name = "Alpha")
        base_model = self.generator_blocks[0]
        
        out = base_model(inp)
        for i in range(1, step):
            out = self.generator_blocks[i](out)

        new_output = self.generator_blocks[step](out)
        new_output = self.to_rgb_blocks[step](new_output)

        old_output = self.to_rgb_blocks[step - 1](out)
        old_output = tf.keras.layers.UpSampling2D()(old_output)

        out = Fade_In()(old_output, new_output, alpha)
        return tf.keras.models.Model([inp, alpha], out, name = "%i_%i_Fade_In_Generator"%(img_res, img_res))




    def Discriminator_Base(self):
        inp = tf.keras.layers.Input(shape = (4, 4, 128), name = "Discriminator_Input")

        out = MinibatchStd()(inp)
      
        out = Conv2D(256, 3)(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = Conv2D(256, 4)(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = tf.keras.layers.Flatten()(out)
        out = Dense(1, 1./8)(out)

        return tf.keras.models.Model(inp, out, name = "Discriminator_Base")

    def Build_From_RGB(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = "Discriminator_Input_%ix%i"%(img_res, img_res))
        out = Conv2D(128, 1)(inp)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        return tf.keras.models.Model(inp, out, name = "From_RGB_%ix%i"%(img_res, img_res))
    
    def Build_Discriminator_Block(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 128))

        out = Conv2D(128, 3, name = "Conv_1_%i%i"%(img_res, img_res))(inp)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = Conv2D(128, 3, name = "Conv_2_%i_%i"%(img_res, img_res))(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        
        out = tf.keras.layers.AveragePooling2D()(out)

        return tf.keras.models.Model(inp, out, name = "%i_%i_Disc_Block"%(img_res, img_res))
    

    def Build_All_Discriminator_And_From_RGB_Blocks(self, n_steps):
        base_model = self.Discriminator_Base()
        from_rgb = self.Build_From_RGB(0)
        self.discriminator_blocks.append(base_model)
        self.from_rgb_blocks.append(from_rgb)
        for i in range(1, n_steps):
            block = self.Build_Discriminator_Block(i)
            from_rgb = self.Build_From_RGB(i)
            self.discriminator_blocks.append(block)
            self.from_rgb_blocks.append(from_rgb)

        
    def Build_Standart_Discriminator(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        from_rgb = self.from_rgb_blocks[step]
        base_model = self.discriminator_blocks[0]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = "Standart_Discriminator_Input")
        alpha = tf.keras.layers.Input(shape = (1), name = "Alpha")
        out = from_rgb(inp)
        for i in range(step, 0, -1):
            out = self.discriminator_blocks[i](out)
        out = base_model(out)
        return tf.keras.models.Model([inp, alpha], out, name = "%ix%i_Standart_Discriminator"%(img_res, img_res))

    def Build_Fade_In_Discriminator(self, step):
        assert step >= 1
        img_res = N_STEP_TO_IMAGE_DIM[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = "Fade_In_Discriminator_Input")
        alpha = tf.keras.layers.Input(shape = (1), name = "Alpha")

        new_output = self.from_rgb_blocks[step](inp)
        new_output = self.discriminator_blocks[step](new_output)

        old_output = tf.keras.layers.AveragePooling2D()(inp)
        old_output = self.from_rgb_blocks[step - 1](old_output)

        out = Fade_In()(old_output, new_output, alpha)

        for i in range(step - 1, 0, -1):
            out = self.discriminator_blocks[i](out)
        out = self.discriminator_blocks[0](out)
        return tf.keras.models.Model([inp, alpha], out, name = "Fade_In_Discriminator_%i_%i"%(img_res, img_res))




    def compile(self):
        super(Progressive_Gan, self).compile()
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-3, 0.0, 0.99, 1e-8, name = "disc_optimizer")
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-3, 0.0, 0.99, 1e-8, name = "gen_optimizer")
        self.disc_loss_metric = tf.keras.metrics.Mean(name = "disc_loss")
        self.gen_loss_metric = tf.keras.metrics.Mean(name = "gen_loss")

    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        real_labels = tf.ones((batch_size, 1), dtype = tf.float32)
        fake_labels = -tf.ones((batch_size, 1), dtype = tf.float32)

        latent_vector = tf.random.normal((batch_size, self.latent_dim))        
        fake_images = self.generator([latent_vector, self.alpha])
        with tf.GradientTape() as gradient_tape,\
                tf.GradientTape() as total_tape:
    
            # forward pass
            pred_fake = self.discriminator([fake_images, self.alpha])
            pred_real = self.discriminator([real_imgs, self.alpha])
                
            epsilon = tf.random.uniform((batch_size,1,1,1))
            interpolates = epsilon*real_imgs + (1-epsilon)*fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator([interpolates, self.alpha])
            
            # calculate losses
            loss_fake = wasserstain_loss(fake_labels, pred_fake)
            loss_real = wasserstain_loss(real_labels, pred_real)
            loss_fake_grad = wasserstain_loss(fake_labels, pred_fake_grad)
            
            
            # gradient penalty      
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.gradient_loss(gradients_fake)
    
            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = 0.001 * tf.reduce_mean(all_pred**2)
            
            disc_loss = loss_fake + loss_real + gradient_penalty + drift_loss
            
            # apply gradients
            gradients = total_tape.gradient(disc_loss, self.discriminator.variables)
            
            self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.variables))

        latent_vector = tf.random.normal((batch_size, self.latent_dim)) 
        with tf.GradientTape() as tape:
            gen_imgs = self.generator([latent_vector, self.alpha])
            pred = self.discriminator([gen_imgs, self.alpha])
            gen_loss = wasserstain_loss(real_labels, pred)
            gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradient, self.generator.trainable_variables))
        self.alpha.assign_add(self.step_per_batch)
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(gen_loss)

        return {"disc_loss" : self.disc_loss_metric.result(), "gen_loss": self.gen_loss_metric.result()}

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1,len(loss.shape)))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        self.penalty_const = 10
        loss = self.penalty_const * loss
        return loss

    def Generate_Models(self, step):
        if not (self.is_in_transition):
            self.generator = self.Build_Standart_Generator(step)
            self.discriminator = self.Build_Standart_Discriminator(step)
        else:
            self.generator = self.Build_Fade_In_Generator(step)
            self.discriminator = self.Build_Fade_In_Discriminator(step)     
        self.compile() 

    def Save_Images(self, n_images, epoch, img_res):
        if (not self.is_in_transition):
            inp = tf.random.normal(shape = (n_images, self.latent_dim))
            out = self.generator([inp, self.alpha])
            out += 1.0
            out *= 255.0
            out = tf.clip_by_value(out, 0.0, 255.0)
            out = tf.cast(out, tf.uint8)
            for i in range(len(out)):
                plt.imsave("Gen_Img/Image%i_%i_%i.jpg"%(img_res, epoch, i), out[i].numpy())

    def Save_Models(self, step):
        if not (self.is_in_transition):
            img_res = N_STEP_TO_IMAGE_DIM[step]
            self.generator.save_weights("Generator_Models/%i_%i"%(img_res, img_res))
            self.discriminator.save_weights("Discriminator_Models/%i_%i"%(img_res, img_res))

    def Load_Models(self, step):
        img_res = N_STEP_TO_IMAGE_DIM[step]
        self.generator.load_weights("Generator_Models/%i_%i"%(img_res, img_res))
        self.discriminator.load_weights("Discriminator_Models/%i_%i"%(img_res, img_res))
            

    def Train(self, step_left_at = None):
        if (step_left_at is None):
            self.Generate_Models(0)
            ds = GenerateDataset(N_IMAGES, (4, 4), BATCH_SIZES[4])
            self.fit(ds, epochs = EPOCHS[4], callbacks = [GAN_Callback(4)])
            self.Save_Models(0)
            for i in range(1, self.n_steps):
                self.alpha.assign(0.0)
                img_res = N_STEP_TO_IMAGE_DIM[i]
                epochs = EPOCHS[img_res]
                batch_size = BATCH_SIZES[img_res]
                n_batches = N_IMAGES // batch_size
                self.step_per_batch = 1.0 / (n_batches * epochs)
                print("\n\n\n%ix%i Models Training"%(img_res, img_res))
                ## Fade_In
                print("\n\nFade_In_Model")
                self.is_in_transition = True 
                self.Generate_Models(i)
                ds = GenerateDataset(N_IMAGES, (img_res, img_res), BATCH_SIZES[img_res])
                self.fit(ds, epochs = EPOCHS[img_res], callbacks = [GAN_Callback(img_res)])
                ## Standart
                print("\n Standart_Model")
                self.is_in_transition = False 
                self.Generate_Models(i)
                self.fit(ds, epochs = EPOCHS[img_res], callbacks = [GAN_Callback(img_res)])
                self.Save_Models(i)
        else:
            self.Generate_Models(step_left_at)
            self.Load_Models()
            img_res = N_STEP_TO_IMAGE_DIM(step_left_at)
            ds = GenerateDataset(N_IMAGES, (img_res, img_res), BATCH_SIZES[img_res])
            self.fit(ds, epochs = EPOCHS[img_res], callbacks = [GAN_Callback(img_res)])
            self.Save_Models(step_left_at)
            for i in range(step_left_at+1, self.n_steps):
                self.alpha.assign(0.0)
                img_res = N_STEP_TO_IMAGE_DIM[i]
                epochs = EPOCHS[img_res]
                batch_size = BATCH_SIZES[img_res]
                n_batches = N_IMAGES // batch_size
                self.step_per_batch = 1.0 / (n_batches * epochs)
                print("\n\n\n%ix%i Models Training"%(img_res, img_res))
                ## Fade_In
                print("\n\nFade_In_Model")
                self.is_in_transition = True 
                self.Generate_Models(i)
                ds = GenerateDataset(N_IMAGES, (img_res, img_res), BATCH_SIZES[img_res])
                self.fit(ds, epochs = EPOCHS[img_res], callbacks = [GAN_Callback(img_res)])
                ## Standart
                print("\n Standart_Model")
                self.is_in_transition = False 
                self.Generate_Models(i)
                self.fit(ds, epochs = EPOCHS[img_res], callbacks = [GAN_Callback(img_res)])
                self.Save_Models(i)


class GAN_Callback(tf.keras.callbacks.Callback):
    def __init__(self, img_res):
        super().__init__() 
        self.img_res = img_res
        self.step = IMAGE_DIM_TO_N_STEP[img_res]
    def on_epoch_end(self, epoch, logs=None):
        self.model.Save_Images(3, epoch, self.img_res)
        self.model.Save_Models(self.step)

    

if __name__ == "__main__":
    prog_gan = Progressive_Gan(LATENT_DIM, 0.2, N_STEPS)
    prog_gan.Train()
    