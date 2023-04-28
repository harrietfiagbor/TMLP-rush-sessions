import gradio as gr  # use pip install gradio to install
import numpy as np  # use pip install numpy to install
from scipy import spatial

# create a function that takes two images as input and returns a similarity score


def cosine_similarity(x, y):
    ''' 
    Resizes the images to 224x224 and flattens them into a list of numbers.
    Returns the cosine similarity between two images.
    cosine similarity value is between 0 and 1, where 0 means the images are completely different and 1 means they are identical.

    '''
    # resize the images to 224x224 and flatten them into a list of numbers
    x = np.resize(x, (224, 224)).flatten()
    y = np.resize(y, (224, 224)).flatten()
    # calculate the cosine similarity between the two images
    similarity = -1 * (spatial.distance.cosine(x, y) - 1)
    # return the similarity score as a string with 2 decimal places
    return f"{similarity:.2f}"


input_1 = gr.inputs.Image()  # accepts an image
input_2 = gr.inputs.Image()  # accepts an image
output = gr.outputs.Textbox()  # returns a text box

# `cosine_similarity` is the function that will be called when the user clicks "Run"
# 2 images are passed to the function as inputs
app = gr.Interface(cosine_similarity, inputs=[
                   input_1, input_2], outputs=output, title="Cosine Similarity", description="Compare two images")
# launch the app
app.launch(share=True)
