import tensorflow as tf
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


def read_image(image_file_name, color_mode=None):
    """
     Read image from file.
    :param image_file_name: full file path
    :param image_size_row_col: [row, col]
    :param color_mode: 'gray', 'rgb' or 'idx'
    :return: numpy image
    """
    img = Image.open(image_file_name.numpy().rstrip())
    if color_mode is not None:
        color_mode = color_mode.numpy()
        if color_mode.lower() == 'gray':
            img = img.convert('L')
        else:
            if color_mode.lower() == 'rgb':
                img = img.convert('RGB')
            else:
                if color_mode.lower() == 'idx':
                    img = img.convert('P')
    return np.array(img)


def tf_read_image(image_file_name, color_model='idx'):
    img = tf.py_function(read_image, [image_file_name, color_model], [tf.float32])
    return tf.transpose(img, [1, 2, 0])


def read_file_list(list_txt_file):
    fp = open(list_txt_file, 'r')
    files = fp.readlines()
    files = [item.rstrip() for item in files]
    return files


def shuffle_lists(*file_lists):
    """
    Example:
        list1=['a', 'b', 'c']
        list2=[1, 2, 3]
        list1_s,list2_s=shuffle_data_files(list1,list2)
        list1_s = ['a', 'c', 'b']
        list2_s = [1, 3, 2]
    :param file_lists: any numbers of list
    :return: shuffled lists
    """
    if len(file_lists) == 1:
        list_files = list(*file_lists)
        np.random.shuffle(list_files)
        return list_files
    else:
        list_files = list(zip(*file_lists))
        np.random.shuffle(list_files)
        return zip(*list_files)


def create_dataset_from_imgs(filenames, labelname, batch_size=2, img_height=400, img_width=400):
    # filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
    # labels = ["/var/data/label1.png", "/var/data/label2.png", ...]

    # read image file
    def _tf_load_image(filename, labelname):
        out_image = tf_read_image(filename, 'gray')
        out_image = tf.image.per_image_standardization(out_image)
        out_label = tf_read_image(labelname, 'idx')
        return out_image, out_label

    #  random crop image
    def _tf_random_crop(image_decoded, label_decoded, out_size=(img_height, img_width, 1)):
        stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, *out_size])
        return cropped_image[0], cropped_image[1]

    # random flip left-right
    def _tf_random_flip_left_right(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_left_right(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    # random flip up-down
    def _tf_random_flip_up_down(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_up_down(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    def _tf_random_transpose(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.transpose(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    def _tf_one_hot(image_decoded, label_decoded):
        convert_label = tf.cast((label_decoded), dtype=tf.uint8)
        one_hot_img = tf.one_hot(convert_label, 4)
        one_hot_label = tf.squeeze(one_hot_img)
        return image_decoded, one_hot_label

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames), tf.convert_to_tensor(labelname)))
    dataset = dataset.map(_tf_load_image)
    dataset = dataset.map(_tf_random_crop)
    dataset = dataset.map(_tf_random_transpose)
    dataset = dataset.map(_tf_random_flip_up_down)
    dataset = dataset.map(_tf_random_flip_left_right)
    dataset = dataset.map(_tf_one_hot)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def example_create_dataset_from_npy(npy_file_path):
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load(npy_file_path) as data:
        features = data["features"]
        labels = data["labels"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset


def example_create_dataset_from_tfrecord(tfrecord_files):
    """"
    Creates a dataset that reads all of the examples from two files.
    training_dataset = example_create_dataset_from_tfrecord(["/training1.tfrecord", "/training1.tfrecord"])
    validation_dataset = example_create_dataset_from_tfrecord(["/validation1.tfrecord", "/validation1.tfrecord"])
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(...)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)
    return dataset


if __name__ == '__main__':
    train_images1 = r'D:\_Artery_Vein_Differentiation\Paper_Data_V3\Train_angio\_list.txt'
    train_labels_lbl = r'D:\_Artery_Vein_Differentiation\Paper_Data_V3\Train_label\_list.txt'

    train_vol_list = read_file_list(train_images1)
    train_lbl_list = read_file_list(train_labels_lbl)

    training_bscan, training_label = shuffle_lists(train_vol_list, train_lbl_list)

    dataset = create_dataset_from_imgs(training_bscan, training_label, batch_size=1)

    sample = dataset.take(1)
    for image, label in dataset.take(6):
        print("Image shape: ", image.numpy().shape)
        print("Label shape: ", label.numpy().shape)
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(np.squeeze(image[:, :].numpy()), cmap='gray')
        plt.subplot(122)
        plt.imshow(np.squeeze(label[:, :].numpy()), cmap='gray')
        plt.show()
