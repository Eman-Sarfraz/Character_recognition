Character Recognition model made by Eman Sarfraz using CNN and ResNet50

Certainly! Here's a combined and detailed explanation of the steps for the character recognition model that utilizes both a custom CNN and a pre-trained ResNet50 model:

### **1. Import Necessary Libraries**
   - **TensorFlow/Keras**: Used for building, training, and evaluating the neural network models.
   - **Pandas and NumPy**: For data manipulation and handling.
   - **OpenCV**: For image processing tasks such as reading, resizing, and normalizing images.
   - **Sklearn**: For splitting the data into training and testing sets, and for encoding labels.

### **2. Load and Preprocess Data**
   - **Read the CSV File**:
     - Load image filenames and their corresponding labels from a CSV file.
   - **Process Images**:
     - Images are read from the specified directory in grayscale mode.
     - Each image is resized to 32x32 pixels for consistency.
     - The pixel values of the images are normalized to the range [0, 1] to improve model performance.
   - **Store and Prepare Data**:
     - Store the processed images in an array with the shape `(32, 32, 1)` for grayscale.
     - Convert labels from text to numerical values using `LabelEncoder`.

### **3. Split Data into Training and Testing Sets**
   - **Train-Test Split**:
     - The dataset is divided into training (80%) and testing (20%) sets using `train_test_split`.

### **4. Data Augmentation**
   - **Set Up Image Augmentation**:
     - `ImageDataGenerator` is used to apply random transformations such as rotations, shifts, and zooms to the training images. This increases the diversity of the training data, which can help the model generalize better.

### **5. Model 1: Custom Convolutional Neural Network (CNN)**
   - **Build the CNN Model**:
     - **Convolutional Layers**: Multiple layers with filters are used to extract features from the input images.
     - **Batch Normalization**: Applied after some layers to stabilize and accelerate the training.
     - **Max Pooling and Dropout**: Pooling layers reduce the spatial dimensions, and dropout layers prevent overfitting by randomly setting some neurons to zero.
     - **Flatten and Dense Layers**: The output from the convolutional layers is flattened and passed through fully connected layers, leading to a softmax layer for classification.
   - **Compile the CNN Model**:
     - The model is compiled using the Adam optimizer, with a learning rate of 0.001.
     - `sparse_categorical_crossentropy` is used as the loss function, and accuracy is tracked as a performance metric.
   - **Train the CNN Model**:
     - The model is trained for 50 epochs using the augmented training data, with validation on the test data.
     - A learning rate scheduler (`ReduceLROnPlateau`) is used to reduce the learning rate if the validation loss does not improve after a certain number of epochs.
   - **Evaluate the CNN Model**:
     - The model's accuracy on the test data is evaluated and printed.

### **6. Model 2: Pre-trained ResNet50**
   - **Prepare Data for ResNet50**:
     - The grayscale images are converted to RGB by duplicating the single grayscale channel into three channels. This is necessary because the ResNet50 model expects three-channel (RGB) input.
   - **Load Pre-trained ResNet50 Model**:
     - The ResNet50 model is loaded with pre-trained weights from ImageNet. The top (fully connected) layers are excluded, allowing us to add custom layers for our specific task.
   - **Add Custom Layers to ResNet50**:
     - **Flatten**: The output of ResNet50 is flattened into a 1D vector.
     - **Dense Layers**: A dense layer with 128 units and ReLU activation is added, followed by a softmax layer with a number of units equal to the number of character classes for classification.
   - **Compile the ResNet50-based Model**:
     - The model is compiled with the Adam optimizer (learning rate = 0.001), and `sparse_categorical_crossentropy` is used as the loss function.
   - **Train the ResNet50-based Model**:
     - The model is trained using the augmented RGB training data. The learning rate scheduler is again used to adjust the learning rate if the validation loss does not improve.
   - **Evaluate the ResNet50-based Model**:
     - The accuracy of the model on the test data is evaluated and printed.

### **7. Visualize Training History**
   - **Plot Training and Validation Metrics**:
     - The training history is plotted, showing both accuracy and loss for the training and validation sets over the epochs. This helps in analyzing the model's performance and understanding how well it generalizes.

### **8. Summary and Discussion**
   - **Combining the Two Models**:
     - The code first trains a custom CNN model and then applies transfer learning using the pre-trained ResNet50 model. Both models are evaluated for their performance on the character recognition task.
   - **Model Performance**:
     - The custom CNN and the ResNet50-based models are both evaluated, with the best model achieving an accuracy of **82%** on the test data.
   - **Character Recognition Task**:
     - The overall goal of this pipeline is to build a model capable of recognizing characters from images. By using both a custom CNN and a pre-trained ResNet50, the model leverages both handcrafted features (from CNN) and powerful pre-trained features (from ResNet50), leading to improved accuracy.

This approach demonstrates the effectiveness of combining custom models with transfer learning techniques in character recognition tasks, resulting in a robust solution with an accuracy of 82%.
