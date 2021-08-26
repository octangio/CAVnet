from CAVnet_model import cavnet
from my_tensorboard import MyTensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from data_process import create_dataset_from_imgs, read_file_list
from tensorflow.keras.optimizers import Adam
from my_model import my_load_model
from my_loss import dice_coef, CE_DL_loss
import argparse

parser = argparse.ArgumentParser(description='tensorflow implementation of CAVnet')
parser.add_argument('--train_images', type=str, default=0,
                    help='Path of training input')
parser.add_argument('--train_labels', type=str, default=0,
                    help='Path of training label')
parser.add_argument('--valid_images', type=str, default=0,
                    help='Path of validation input')
parser.add_argument('--valid_labels', type=str, default=0,
                    help='Path of validation label')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size')
parser.add_argument('--input_height', type=int, default=2,
                    help='the height of input image')
parser.add_argument('--input_width', type=int, default=2,
                    help='the width of input image')


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    input_height = args.input_height
    input_width = args.input_width
    img_size = (input_height, input_width, 1)
    epochs = 5000

    # load network structure
    model = cavnet(input_shape=img_size)
    model.compile(optimizer=Adam(lr=0.001), loss=CE_DL_loss, metrics=[dice_coef, 'acc'])
    print(model.summary())

    # read data list
    train_img_list = read_file_list(args.train_images)
    train_lbl_list = read_file_list(args.train_labels)

    valid_img_list = read_file_list(args.valid_images)
    valid_lbl_list = read_file_list(args.valid_labels)

    # create data generator
    my_training_batch_generator = create_dataset_from_imgs(train_img_list, train_lbl_list, batch_size,
                                                           img_height=input_height, img_width=input_width)
    my_validation_batch_generator = create_dataset_from_imgs(valid_img_list, valid_lbl_list, batch_size,
                                                             img_height=input_height, img_width=input_width)
    logdir = r'logs'
    # callbacks
    tensorboard_visualization = MyTensorBoard(log_dir=logdir,
                                              write_graph=True, write_images=True)
    csv_logger = CSVLogger('{}/training.log'.format(logdir))
    checkpoint = ModelCheckpoint(logdir + '/model_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True)
    resuce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0.00000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    callbacks = [tensorboard_visualization, checkpoint, resuce_lr, early_stopping, csv_logger]
    [model, init_epoch] = my_load_model(model, logdir=logdir, checkpoint_file='checkpoint.ckp',
                                        custom_objects={})
    # training
    model.fit_generator(generator=my_training_batch_generator,
                        epochs=epochs,
                        initial_epoch=init_epoch,
                        steps_per_epoch=200,
                        validation_steps=50,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=my_validation_batch_generator)
