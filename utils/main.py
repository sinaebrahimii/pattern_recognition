import  cv2
from cv2 import typing
import numpy as np
from scipy.ndimage import  convolve


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
   
def lbp(image):
    height, width, _ = image.shape 
    img_gray = cv2.cvtColor(image, 
                        cv2.COLOR_BGR2GRAY) 
    img_lbp = np.zeros((height, width), 
                   np.uint8)
    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp


def lpq(image, win_size=3):
    
    STFTalpha = 1 / win_size
    conv_mode = 'reflect'
    
    x = np.arange(-(win_size - 1) // 2, (win_size - 1) // 2 + 1, dtype=float)
    n = len(x)
    w0 = np.ones_like(x) / np.sqrt(n)

    w1 = np.exp(-2 * np.pi * 1j * x * STFTalpha)
    
    f = np.zeros((n, n), dtype=complex)

    f[:, 0] = w0
    f[:, 1] = w1
    f[:, 2] = np.conj(w1)
    
    img = image.astype('float32')
    f_img = np.zeros((3, img.shape[0], img.shape[1]), dtype=complex)

    for i in range(3):
        f_img[i] = convolve(img, np.real(f[:, i].reshape(n, 1)), mode=conv_mode) + \
                   1j * convolve(img, np.imag(f[:, i].reshape(n, 1)), mode=conv_mode)

    # Create LPQ descriptor by thresholding the real and imaginary parts
    lpq_desc = ((np.angle(f_img[0]) > 0).astype(int) +
                2 * (np.angle(f_img[1]) > 0).astype(int) +
                4 * (np.angle(f_img[2]) > 0).astype(int)).astype(np.uint8)
    
    return lpq_desc


