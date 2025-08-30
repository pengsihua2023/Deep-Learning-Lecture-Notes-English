## Famous network: ResNet(2015): Residual Network
- **Proposed by**: Kaiming He (Microsoft Research)
<div align="center">
  <img width="220" height="220" alt="image" src="https://github.com/user-attachments/assets/005ed077-913c-4432-9b16-6e2678626fa7" />
</div>

- **Features**: Introduces residual connections to address the vanishing gradient problem in deep networks, enabling the construction of networks with hundreds of layers.  
- **Key Points to Master**: Residual learning, deep network training techniques.  
- **Importance**:  
ResNet is an advanced version of CNNs that uses "residual connections" to mitigate the vanishing gradient problem, allowing the construction of very deep networks (tens to hundreds of layers).  
It excels in image classification tasks (e.g., ImageNet competition) and serves as a cornerstone of modern computer vision.  
- **Core Concept**:  
ResNet introduces "residual connections" (skip connections), enabling the network to learn "changes" rather than direct outputs, reducing the training difficulty of deep networks.  
- **Applications**: Image classification, object detection (e.g., object recognition in autonomous driving), facial recognition.
<div align="center">
<img width="400" height="260" alt="image" src="https://github.com/user-attachments/assets/4c111489-898f-4412-9e70-336ec2320f03" />  
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/ee800edc-db6e-4cde-84d9-0d396ca69e58" />  
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


## Code

This code implements a **simplified ResNet model** for the image classification task on the **CIFAR-10 dataset**. The main functionalities are as follows:

1. **Residual Block Definition**:
   - Implements the `residual_block` function, defining a residual module with two convolutional layers (with batch normalization and ReLU activation) and a residual connection.
   - Supports dimension adjustment (via 1x1 convolution in the shortcut branch) to ensure input and output dimensions match.

2. **Model Construction**:
   - Defines the `build_simple_resnet` function to construct a simplified ResNet model:
     - Initial convolutional layer (64 filters, 3x3 convolution).
     - Stacks 4 residual blocks (two groups with 64 channels and two with 128 channels, with the second group using stride=2 for downsampling).
     - Global average pooling followed by a fully connected layer, outputting 10-class classification results (with softmax activation).

3. **Data Preprocessing**:
   - Loads the CIFAR-10 dataset (32x32 color images).
   - Normalizes pixel values to the [0,1] range.

4. **Model Compilation and Training**:
   - Compiles the model using the Adam optimizer and sparse categorical cross-entropy loss function, tracking accuracy metrics.
   - Trains the model for 10 epochs with a batch size of 64, using the test set for validation.

The code is implemented using TensorFlow/Keras, suitable for CIFAR-10 image classification, outputs a model summary, and performs training to learn image classification features.

## Code

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters, kernel_size=3, stride=1):
    """Define a simple residual block"""
    shortcut = x
    
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut dimensions if they don't match
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Residual connection
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_simple_resnet(input_shape=(32, 32, 3), num_classes=10):
    """Build a simple ResNet model"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stack residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x)
    return model

# Example: Create and compile the model
if __name__ == "__main__":
    model = build_simple_resnet()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()
    
    # Assume testing with CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Data preprocessing
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

```
