from keras_squeezenet import SqueezeNet
from DataGen2 import ImageDataGeneratorV2
from keras.optimizers import Adam
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Flatten
import tensorflow as tf
from keras.regularizers import l2
import keras.backend as kb
import keras.losses as losses


def custom_mse_lm(y_true,y_pred):
    return kb.sign(kb.sum(kb.abs(y_true),axis=-1))*kb.sum(kb.square(tf.multiply((kb.sign(y_true)+1)*0.5, y_true-y_pred)),axis=-1)/kb.sum((kb.sign(y_true)+1)*0.5,axis=-1)

def custom_mse_pose(y_true,y_pred):
    return kb.sign(kb.sum(kb.abs(y_true),axis=-1))*losses.mean_squared_error(y_true,y_pred)


def get_RCNN_face_model():
    num_outputs = 2
    print "Building SqueezeNet..."
    model = SqueezeNet()
    print "Building New Layers..."
    x = model.get_layer('drop9').output
    x = Convolution2D(num_outputs, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    model = Model(inputs=model.input, outputs=out)
    print "Model has been generated"
    return model





model = get_RCNN_face_model()
model.load_weights('RCNN_face.h5')


drop9_out = model.get_layer('drop9').output
drop9_flat = Flatten(name='flat')(drop9_out)





landmark_FC = Dense(500, name='landmark_FC', activation='relu', init='he_normal', W_regularizer=l2(0.00))(drop9_flat)
landmark_out = Dense(42, name='landmark_out', activation='sigmoid', init='he_normal', W_regularizer=l2(0.00))(landmark_FC)

visibility_FC = Dense(500, name='visibility_FC', activation='relu', init='he_normal', W_regularizer=l2(0.00))(drop9_flat)
visibility_out = Dense(21, name='visibility_out', activation='sigmoid', init='he_normal', W_regularizer=l2(0.00))(visibility_FC)

pose_FC = Dense(500, name='pose_FC', activation='relu', init='he_normal', W_regularizer=l2(0.00))(drop9_flat)
pose_out = Dense(3, name='pose_out', activation='tanh', init='he_normal', W_regularizer=l2(0.00))(pose_FC)

gender_conv = Convolution2D(2, (1, 1), padding='valid', name='gender_conv')(drop9_out)
gender_conv = Activation('relu', name='relu_gender_conv')(gender_conv)
gender_conv = GlobalAveragePooling2D()(gender_conv)
gender_out = Activation('softmax', name='gender_out')(gender_conv)

fnf_conv = Convolution2D(2, (1, 1), padding='valid', name='fnf_conv')(drop9_out)
fnf_conv = Activation('relu', name='relu_fnf_conv')(fnf_conv)
fnf_conv = GlobalAveragePooling2D()(fnf_conv)
fnf_out = Activation('softmax', name='fnf_out')(fnf_conv)


model = Model(inputs=model.input, outputs=[fnf_out,landmark_out, visibility_out, pose_out, gender_out])

model.load_weights('multiout.h5')

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,
              loss={'landmark_out': custom_mse_lm, 'visibility_out': custom_mse_lm, 'pose_out': custom_mse_pose, 'fnf_out': 'categorical_crossentropy', 'gender_out': 'categorical_crossentropy'},
              loss_weights={'landmark_out': 1, 'visibility_out': 1, 'pose_out': 1, 'fnf_out': 1, 'gender_out': 1})

#model.compile(optimizer=optimizer,
#              loss={'landmark_out': custom_mse_lm, 'visibility_out': custom_mse_lm, 'pose_out': custom_mse_pose, 'fnf_out': 'categorical_crossentropy', 'gender_out': 'categorical_crossentropy'},
#              loss_weights={'landmark_out': 0, 'visibility_out': 0, 'pose_out': 1, 'fnf_out': 0, 'gender_out': 0})


print "Building Train Data Generator..."
train_data = ImageDataGeneratorV2()
train_data_flow = train_data.flow_from_directory('flickr','trainP_new.json', 'trainN.json', output_type='hyperface', pos_batch_size=64, neg_batch_size=64)
#print "Building Validation Data Generator..."
#val_data = ImageDataGeneratorV2()
#val_data_flow = val_data.flow_from_directory('flickr', 'val.json', 'val.json', output_type='hyperface', pos_batch_size=64, neg_batch_size=0)


# checkpoint
filepath="weights-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='train_loss', verbose=1, save_best_only=False, mode='min', period=40)
callbacks_list = [checkpoint]

print "Start Training..."
output_file_name = 'multiout'+'.h5'
model.fit_generator(train_data_flow, steps_per_epoch=100, epochs=300, callbacks=callbacks_list)
model.save(output_file_name)
