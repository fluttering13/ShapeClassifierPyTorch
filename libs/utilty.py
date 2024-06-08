import os
from PIL import Image
from IPython.display import display, HTML
import io
import base64
def image_save(image,i,save_path,shape_type):#save image fn
    if not os.path.exists(save_path):
        os.mkdir(save_path)    
    if not os.path.exists(save_path+shape_type):
        os.mkdir(save_path+shape_type)
    image.save(save_path+shape_type+'/'+str(i)+'.png') 

def show_image(image_path):#show image with Jupyter display
    # 加载图像
    image = Image.open(image_path)
    # 显示图像
    display(image)


def display_side_by_side(images):#show a lot of image with Jupyter display
    def image_to_base64(image):


        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
        return image_base64


    html_str = ""
    for image_path in images:

        image = Image.open(image_path)

        image = image.resize((200,200))

        image_base64 = image_to_base64(image)

        html_str += f'<img src="data:image/png;base64,{image_base64}" style="display:inline-block;margin:10px;"/>'
    
    display(HTML(html_str)) 