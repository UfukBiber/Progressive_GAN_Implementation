import tensorflow as tf 
from Custom_Layers import wasserstain_loss

def Generate_Gan_Models(discriminators, generators):
    models = list()
    for i in range(len(discriminators)):
        ## Standart Gan
        discriminator = discriminators[i][0]
        discriminator.trainable = False 
        generator = generators[i][1]
        standart_gan_model = tf.keras.Sequential([generator,discriminator])
        standart_gan_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3, 0., 0.999, 1e-8), loss = wasserstain_loss)
        ## Fade in Gan

        discriminator = discriminators[i][1]
        discriminator.trainable = False 
        generator = generators[i][1]
        fade_in_gan_model= tf.keras.Sequential([generator, discriminator])
        fade_in_gan_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3, 0., 0.999, 1e-8), loss = wasserstain_loss)
        models.append([standart_gan_model, fade_in_gan_model])
    return models



