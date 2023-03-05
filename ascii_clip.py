
# A simple, fun project to 'evolve' a picture using only ASCII characters, 
# and try to make CLIP think it corresponds to a given text. 

# The idea is to use a genetic algorithm to evolve a picture, pixel by pixel,

# Let's load some libraries
import numpy as np
from skimage import color, io
import random
from PIL import ImageFont, ImageDraw, Image


# Define the ascii parameters
text_width = 40 # the number of characters per line
text_height = 20 # the number of lines
ascii_chars = list('o*0/\\|-+~.')
# Let's start simple. Just full or empty pixels!
# ascii_chars = [' ', 'o']
imsize = [224,224]
text_size = 5
text_prompt = "A black circle on a white background"
out_dir = './out/'

# Get a font only once
fnt = ImageFont.truetype('FreeMono.ttf', 15)

# Define a text rendering algorithm
def render_text(text, imsize, draw=False):
    # Create a new image
    img = Image.new('RGB', imsize, color = (255, 255, 255))
    # Get a drawing context
    d = ImageDraw.Draw(img)
    # Draw text, half opacity
    d.text((text_size,text_size), text, font=fnt, fill=(0,0,0,128))
    # Save the image
    if draw:
        img.save(out_dir+draw+'.png')
    return img

# Just a debug.... make a random field of ascii characters
def random_ascii(text_width, text_height, ascii_chars):
    # this has to be as fast as possible - use numpy random with replacement. 
    random_strings = "\n".join(["".join(np.random.choice(ascii_chars, size=(text_width)))])
    return random_strings


# Non-random ascii, this time from a set of integers
def int_ascii(text_width, text_height, ascii_chars, int_array):
    # Convert the integers to ascii characters
    ascii_strings = [ascii_chars[i] for i in int_array]
    # Split into chunks of text_width
    ascii_strings = [ascii_strings[i:i+text_width] for i in range(0, len(ascii_strings), text_width)]
    # Join the lines
    ascii_strings = "\n".join(["".join(i) for i in ascii_strings])
    return ascii_strings



from time import time
# t1 = time()
# test_text = random_ascii(text_width, text_height, ascii_chars)
# t2 = time()
# render_text(test_text, imsize)
# t3 = time()

# print('Time to generate random ascii: ', t2-t1)
# print('Time to render text: ', t3-t2)

# Now let's define the fitness function - let's load clip
import torch
import torchvision
import clip
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# First embed the text as it's constant
with torch.no_grad():
    text = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text)

# Define a function to calculate the similarity of a given image
def similarity(image):
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        # Calculate the cosine similarity between the image and the text
        similarity = (100.0 * image_features @ text_features.T).cpu().numpy()
        return similarity.squeeze()
    
# Now a fitness function where we do the whole thing
def fitness_function(solution, solution_idx):
    # Convert the solution to ascii
    ascii_string = int_ascii(text_width, text_height, ascii_chars, solution)

    # Render the text
    image = render_text(ascii_string, imsize, )
    # Calculate the similarity
    sim = similarity(image)
    # Return the fitness
    return float(sim)

# # Let's test the fitness function - run 5 runs
# for i in range(5):
#     # Generate a random solution
#     solution = np.random.randint(0, len(ascii_chars), size=(text_height*text_width))
#     # Calculate the fitness
#     fitness = fitness_function(solution, i)
#     print('Fitness: ', fitness)

# Now for the genetic algorithm
# import pygad
# def callback_gen(ga_instance):
#     print("Generation : ", ga_instance.generations_completed)
#     print("Fitness of the best solution :", ga_instance.best_solution()[1])
#     ascii_string = int_ascii(text_width, text_height, ascii_chars, ga_instance.best_solution()[0])
#     image = render_text(ascii_string, imsize, draw='generation_'+str(ga_instance.generations_completed))


# sol_per_pop = 100
# ga_instance = pygad.GA(num_generations=10,
#                        num_parents_mating=5,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=(text_height*text_width),
#                        fitness_func=fitness_function,
#                        init_range_low=0,
#                        init_range_high=len(ascii_chars),
#                        gene_type=int, 
#                        on_generation=callback_gen,
#                        parent_selection_type="sss",
#                        mutation_percent_genes = 20,
#                         #   mutation_type="swap",
#                        )
# ga_instance.run()
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# # Draw the best solution
# ascii_string = int_ascii(text_width, text_height, ascii_chars, solution)
# image = render_text(ascii_string, imsize, draw='best')


# Let's instead try doing backprop with raw pixels. 

image_dim = 224*224

def fitness_function_raw(solution, solution_idx):
    # convert solution to pixels
    pixels = np.array(solution).reshape((224,224,1))*255
    # repeat 3 times to make it 'color'
    pixels = np.repeat(pixels, 3, axis=2)
    # convert to image
    image = Image.fromarray(pixels.astype('uint8'), 'RGB')
    # Calculate the similarity
    sim = similarity(image)
    # Return the fitness
    return float(sim)


layer_tanh = nn.Tanh()
layer_cosine = nn.CosineSimilarity(dim=1)
layer_upsample = nn.UpsamplingNearest2d(size=(224,224)) 

def loss_func(imtensor, printout=False):
    imtensor = layer_tanh(imtensor)
    # imtensor is now between -1 and +1 - let's add 1 and divide by 2 to get between 0 and 1
    imtensor = (imtensor+1)/2
    # imtensor = torchvision.transforms.Resize((imsize,imsize))(imtensor)
    imtensor = layer_upsample(imtensor)
    # repeat 3 times to make it 'color'
    imtensor = torch.repeat_interleave(imtensor, 3, dim=1)
    image_features = model.encode_image(imtensor)
    similarity = layer_cosine(image_features, text_features)

    # Inverse regularisation - we want to maximise the std dev of the image
    # std_dev = torch.std(imtensor)
    std_dev = 0

    loss = -similarity - std_dev

    if printout:
        print('Similarity: {}, stddev: {}, loss: {}'.format(similarity, std_dev, loss))
        pixels =  imtensor.detach().cpu().numpy().squeeze()*255
        # Here the dims are still 3, 224, 224
        # We want 224, 224, 3 
        pixels = np.transpose(pixels, (1,2,0))
        return pixels
    return loss



 

# instead of a genetic algorithm let's try sgd in pytorch

imsize_small = 32
imsize = 224

# define the image tensors
imtensor = torch.rand((1,1,imsize_small,imsize_small), requires_grad=True, device=device)
optimizer = torch.optim.Adam([imtensor], lr=0.3)
# optimizer = torch.optim.SGD([imtensor], lr=0.8, momentum=0.9)

n_iters = 10000

for i in tqdm.trange(n_iters):
    optimizer.zero_grad()
    loss_func(imtensor).backward()
    optimizer.step()
    # every 100 iterations, save the image
    if i%100==0:
        # with torch.no_grad():
            # print loss
        optimizer.zero_grad()
        pixels = loss_func(imtensor, printout=True)
        image = Image.fromarray(pixels.astype('uint8'), 'RGB')
        image.save(out_dir + 'sgd_'+str(i)+'.png')
