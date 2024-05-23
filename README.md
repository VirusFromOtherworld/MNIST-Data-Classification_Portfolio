# MNIST Digit Classifier

This project implements a neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras in Python.

## Dependencies

- Python 3.8 or above
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation

Clone this repository to your local machine:

git clone https://github.com/your-username/your-repository.git
cd your-repository

Install the required packages:

pip install -r requirements.txt

## Usage

Run the script `train_model.py` to train the model:

python train_model.py

This will train the neural network on the MNIST dataset and save the trained model.

## Model Architecture

The model consists of the following layers:

- Flatten Layer: Flattens the input data to a 1D array.
- Dense Layer 1: 128 neurons, ReLU activation.
- Dense Layer 2: 10 neurons, Softmax activation.

The model uses the Adam optimizer and sparse categorical crossentropy as the loss function.

## Results

After training, the model achieves the following performance:

- Accuracy on training data: 99.9%
- Accuracy on test data: 97.5%

## Visualization

You can visualize the results and the model's predictions using:

python visualize_results.py

This will display images from the MNIST dataset along with the model's predictions.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
