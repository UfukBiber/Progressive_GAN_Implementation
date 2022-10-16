import tensorflow as tf 
import os 

IMAGE_PATH = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebA-HQ-img"


def LoadImg(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = img_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img 


def GenerateDataset(n_images, img_size, batch_size):
    paths = [os.path.join(IMAGE_PATH, path) for path in os.listdir(IMAGE_PATH) if path[-3:] == "jpg"]
    paths = paths[:n_images]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda path : LoadImg(path, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds 

# if __name__ == "__main__":
#     ds = GenerateDataset((4, 4))
#     for batch in ds:
#         print(batch)
#         break