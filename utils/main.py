import  cv2
from cv2 import typing
import numpy as np

def make_blocks(img:typing.MatLike,)->list[typing.MatLike]:
    img=cv2.resize(img,(100,100))
    width,height=img.shape[:2]
    block_height = height // 3
    block_width = width // 3

    blocks :list =[]
    for i in range(3):
        for j in range(3):
            start_y = i * block_height
            start_x = j * block_width
            end_y = start_y + block_height
            end_x = start_x + block_width
            block = img[start_y:end_y, start_x:end_x]
            blocks.append(block)
        
    return blocks



def mean_variance(blocks:list[typing.MatLike],show:bool =False)->tuple[list[float],list[float]]:
    means = []
    vars =[]

    for  block in blocks:
        means.append(block.mean())
        vars.append(block.var())
        if show:
            print("Mean:")
            print(block.mean())
            print("Variance:")
            print(block.var())
    return means,vars



def corr(blocks_1:list[typing.MatLike],blocks_2 :list[typing.MatLike]):
    corrs=[]
    for block in range(len(blocks_1)):
        result=cv2.matchTemplate(blocks_1[block],blocks_2[block],cv2.TM_CCOEFF_NORMED)
        corrs.append(result)
    return corrs


def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        # If local neighbourhood pixel  
        # value is greater than or equal 
        # to center pixel values then  
        # set it to 1 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
      
        pass
      
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # convert binary values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 
   
 