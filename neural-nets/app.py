import gradio as gr
import numpy as np
import nn  # neural-nets/nn.py

neural_network = nn.NeuralNet()

# print randomly generated weights before training
print("Beginning Randomly Generated Weights: ")
print(neural_network.weights)

# training taking place
# training data consisting of 4 examples--3 input values and 1 output
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T
# training taking place
neural_network.train(training_inputs, training_outputs, 10000)

# print weights after training
print("Ending Weights After Training: ")
print(neural_network.weights)

# prediction taking place


def predict(input1, input2, input3):
    prediction = neural_network.think(np.array([input1, input2, input3]))
    return prediction[0]


# inputs = [1 0 0]
inputs = [gr.inputs.Number(label=f"Input {i}") for i in range(1, 4)]
# output = [close to 1]
output = gr.outputs.Label()

title = "Neural Network Prediction"
description = "Enter three input values to predict the corresponding output."
gr.Interface(fn=predict, inputs=inputs, outputs=output,
             title=title, description=description).launch(share=True)
