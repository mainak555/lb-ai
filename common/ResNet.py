from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Dense
from keras.layers.merge import add
from keras.regularizers import l2
from keras import Model
#%%
class ResNet:
    @staticmethod
    def Layer(inputs, filters=16, kernel_size=3, strides=1, activation='relu', 
        batch_normalization=True, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
        conv_first=True #True: conv->bn->activation, False: bn->activation->conv
        ): 
        '''2D Convolution-Batch Normalization Activation Stack Builder
            Returns x (tensor): tensor as input to the next layer
        '''
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x        
#################
    @staticmethod
    def V2(input_shape, depth, num_classes):
        '''
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D / bottleneck layer
        '''
        if(depth - 2) % 9 !=0:
            raise ValueError('dept should be 9N+2')
        
        num_filters_in = 16
        num_resedual_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        #v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = ResNet.Layer(inputs=inputs, filters=num_filters_in, conv_first=True) 

        #Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_resedual_blocks):
                activation = 'relu'
                batch_norm = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:          # first layer and first stage
                        activation = None
                        batch_norm = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:      # first layer but not first stage
                        strides = 2         # downsample

                # bottleneck residual unit
                y = ResNet.Layer(inputs=x, filters=num_filters_in, kernel_size=1, strides=strides, 
                    activation=activation, batch_normalization=batch_norm, conv_first=False)
                y = ResNet.Layer(inputs=y, filters=num_filters_in, conv_first=False)
                y = ResNet.Layer(inputs=y, filters=num_filters_out, kernel_size=1, conv_first=False)

                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    x = ResNet.Layer(inputs=x, filters=num_filters_out, kernel_size=1, 
                        strides=strides, activation=None, batch_normalization=False)
                
                x = add([x, y])
#-----------EOF (inner)
            num_filters_in = num_filters_out
#-------EOF (outer)
        # add classifier on top
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

        model = Model(inputs=inputs, outputs=out)
        return model
#################
    @staticmethod
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning Rate: ', lr)
        return lr