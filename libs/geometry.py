import random
import math

def determine_color(color_type):
    if color_type == 'fix':
        color = (0, 255, 0)
    elif color_type == 'random':
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        assert False, 'color_type must be fix or random'       
    return color

def determine_radius_perserving_gemotry(draw, height, width, shape_type, color, side_length_type, rotation_bool):
    
    if shape_type not in ['circle','square']:
        return draw
    
    # set innital point on the center
    center_x=width//2
    center_y=height//2

    x = center_x+random.randint(-int(width/4), int(width/4))
    y = center_y+random.randint(-int(height/4), int(height/4))

    if side_length_type == 'fix':
        radius=min(int(width/4),int(height/4))
    elif side_length_type == 'random':
        bound_max=min(min(width-x,x),min(height-y,y)) # prevent from out of bounbds
        bound_min=bound_max//10 # no way to invisible
        radius = random.randint(bound_min, bound_max)
    else:
        assert False, 'side_length must be fix or random'

    if shape_type == 'circle':
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    elif shape_type== 'square':
        x1=x-radius
        x2=x+radius
        y1=y-radius
        y2=y+radius
        if rotation_bool: 
            roatation_continue=True
            while roatation_continue:
                theta = random.uniform(0, 2 * math.pi/4)
                new_x1 = center_x + (x1 - center_x) * math.cos(theta) - (y1 - center_y) * math.sin(theta)
                new_y1 = center_y + (x1 - center_x) * math.sin(theta) + (y1 - center_y) * math.cos(theta)
                new_x2 = center_x + (x2 - center_x) * math.cos(theta) - (y2 - center_y) * math.sin(theta)
                new_y2 = center_y + (x2 - center_x) * math.sin(theta) + (y2 - center_y) * math.cos(theta)
                x1,x2,y1,y2 = new_x1,new_x2,new_y1,new_y2
                roatation_continue=(x1>width or x2>width or y1>height or y2>height or x1>x2 or y1>y2)
                
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return draw

def determine_triangle(draw, height, width, shape_type, color, side_length_type, rotation_bool):
    if shape_type != 'triangle':
        return draw
    
    if side_length_type=='fix': #equilateral triangle
        side_length = min(width // 3, height // 2)
        rotation_angle = random.randint(0, 360)
        if not rotation_bool:
            rotation_angle = 0
        angle = math.radians(rotation_angle)
        center_x=width//2
        center_y=height//2
        x = center_x+random.randint(-int(width/4), int(width/4))
        y = center_y+random.randint(-int(height/4), int(height/4))                    
        x1 = x
        y1 = y - side_length
        x2 = x + side_length * (3 ** 0.5 / 2) * math.cos(angle) - side_length / 2 * math.sin(angle)
        y2 = y + side_length / 2 * math.cos(angle) + side_length * (3 ** 0.5 / 2) * math.sin(angle)
        x3 = x - side_length * (3 ** 0.5 / 2) * math.cos(angle) + side_length / 2 * math.sin(angle)
        y3 = y + side_length / 2 * math.cos(angle) + side_length * (3 ** 0.5 / 2) * math.sin(angle)
        coordinates=[(x1, y1), (x2, y2), (x3, y3)]

    elif side_length_type == 'random':
        x_list=[random.randint(0, width) for _ in range(3)]
        y_list=[random.randint(0, height) for _ in range(3)]
        coordinates = list(zip(x_list, y_list))
    draw.polygon(coordinates, fill=color)

    return draw