from PIL import Image, ImageDraw
from libs.geometry import *
from libs.utilty import *
def generate_shape_image(width, height, num_each_shape, shape_type, save_path, color_type='random',side_length_type='random',rotation_bool=True):

    #label_dict={}
    for i in range(num_each_shape):
        # Create a blank white image
        image = Image.new("RGB", (width, height), "white")

        draw = ImageDraw.Draw(image)        
        # determine color
        color=determine_color(color_type)
        # determine square or circle
        draw=determine_radius_perserving_gemotry(draw, height, width, shape_type, color, side_length_type, rotation_bool)
        # determine triangle
        draw=determine_triangle(draw, height, width, shape_type, color, side_length_type, rotation_bool)

        image_save(image,i,save_path,shape_type)


if __name__ == "__main__":
    width=200
    hight=200
    num_each_shape=1000
    for shape_type in ['circle','square','triangle']:
        generate_shape_image(width, hight, 
                            num_each_shape, 
                            shape_type,
                            "./pic/",
                            color_type='fix', #random or fix
                            side_length_type='fix', #whether fix the side_length of geometry
                            rotation_bool=True) #whether fix the roation
