import numpy as np

def HSV2img(H_img, S_img, V_img):
    height, width = H_img.shape
    recon_image = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            h_temp = H_img[h][w]/60
            f = h_temp - int(h_temp)
            t = V_img[h][w]*(1-S_img[h][w])
            n = V_img[h][w]*(1-S_img[h][w]*f)
            p = V_img[h][w]*(1-S_img[h][w]*(1-f))
            if int(h_temp)<1:
                r,g,b = V_img[h][w], p, t
            elif int(h_temp<2):
                r,g,b = n, V_img[h][w], t
            elif int(h_temp<3):
                r,g,b = t, V_img[h][w], p
            elif int(h_temp<4):
                r,g,b = t, n, V_img[h][w]
            elif int(h_temp<5):
                r,g,b = p, t, V_img[h][w]
            else:
                r,g,b = V_img[h][w], t, n
            recon_image[h][w][0] = r
            recon_image[h][w][1] = g
            recon_image[h][w][2] = b
    return recon_image

def get_HSV(img):
    '''
    Arg:
        img - 3D np array [height, weight, channel(=3)]
    return:
        H_img, S_img, V_img 
            shape with [height, width]
    '''
    height, width, channel = img.shape
    H_img = np.zeros((height, width))
    S_img = np.zeros((height, width))
    V_img = np.zeros((height, width))    
    for h in range(height):
        for w in range(width):
            R_value = img[h][w][0]
            G_value = img[h][w][1]
            B_value = img[h][w][2]
            M = np.max(img[h,w,:])
            m = np.min(img[h,w,:])
            D = M - m
            if D==0:
                H_img[h][w] = 0
            elif M == R_value:
                 H_img[h][w] = 60*(G_value-B_value)/D
            elif M == G_value:
                 H_img[h][w] = 60*((B_value-R_value)/D+2)
            elif M == B_value:
                 H_img[h][w] = 60*((R_value-G_value)/D+4)
            if H_img[h][w] < 0:
                H_img[h][w]+=360
            if M==0:
                V_img[h][w] = 0
                S_img[h][w] = 0
            else:
                V_img[h][w] = M
                S_img[h][w] = D/V_img[h][w]
    return H_img, S_img, V_img
def get_HSI(img):
    '''
    Arg:
        img - 3D np array [height, weight, channel(=3)]
    return:
        H_img, S_img, L_img 
            shape with [height, width]
    '''
    height, width, channel = img.shape
    H_img = np.zeros((height, width))
    S_img = np.zeros((height, width))
    I_img = np.zeros((height, width))    
    for h in range(height):
        for w in range(width):
            R_value = img[h][w][0]
            G_value = img[h][w][1]
            B_value = img[h][w][2]
            M = np.max(img[h,w,:])
            m = np.min(img[h,w,:])
            D = M - m
            if D==0:
                H_img[h][w] = 0
            elif M == R_value:
                 H_img[h][w] = 60*(G_value-B_value)/D
            elif M == G_value:
                 H_img[h][w] = 60*((B_value-R_value)/D+2)
            elif M == B_value:
                 H_img[h][w] = 60*((R_value-G_value)/D+4)
            if H_img[h][w] < 0:
                H_img[h][w]+=360
            I_img[h][w] = (R_value+G_value+B_value)/3
            if m == 0:
                S_img[h][w] = 0
            else :
                S_img[h][w] = 1-m/I_img[h][w]
    return H_img, S_img, I_img

def get_HSL(img):
    '''
    Arg:
        img - 3D np array [height, weight, channel(=3)]
    return:
        H_img, S_img, L_img 
            shape with [height, width]
    '''
    height, width, channel = img.shape
    H_img = np.zeros((height, width))
    S_img = np.zeros((height, width))
    L_img = np.zeros((height, width))    
    for h in range(height):
        for w in range(width):
            R_value = img[h][w][0]
            G_value = img[h][w][1]
            B_value = img[h][w][2]
            M = np.max(img[h,w,:])
            m = np.min(img[h,w,:])
            D = M - m
            if D==0:
                H_img[h][w] = 0
            elif M == R_value:
                 H_img[h][w] = 60*(G_value-B_value)/D
            elif M == G_value:
                 H_img[h][w] = 60*((B_value-R_value)/D+2)
            elif M == B_value:
                 H_img[h][w] = 60*((R_value-G_value)/D+4)
            if H_img[h][w] < 0:
                H_img[h][w]+=360          
            L_img[h][w] = (M+m)/2
            S_img[h][w] = 1-np.abs(2*L_img[h][w]-1)
    return H_img, S_img, L_img


def clip(x, vmax = 255, vmin = 0):
    '''
        Clip the value of x between vmax(=255), and vmin(=0)
    '''
    if x>vmax:
        return vmax
    if x<vmin:
        return vmin
    return x

def clip_2d(array, vmax = 255, vmin = 0 ): 
    height, width = array.shape
    clipped_array = np.zeros_like(array)
    for h in range(height):
        for w in range(width):
            clipped_array[h][w] = clip(array[h][w], vmax=vmax, vmin=vmin)
    return clipped_array
def cycle_clip(value, turn):
    return value-turn*(int(value)//turn)
def cycle_clip_2d(array, turn): 
    height, width = array.shape
    clipped_array = np.zeros_like(array)
    for h in range(height):
        for w in range(width):
            clipped_array[h][w] = cycle_clip(array[h][w],turn)
    return clipped_array
def conv(img, filt):
    '''
        img : 2D np array
        filt : 2D np array
    '''
    ir, ic = img.shape
    fr, fc = filt.shape
    
    conv_img = np.zeros((ir, ic))
    for r in range(ir):
        for c in range(ic):
            value = 0
            for i in range(fr):
                for j in range(fc):
                    if r+i>=0 and r+i<ir and c+j>=0 and c+j<ic :
                        value += filt[i][j]*img[r+i][c+j]
            conv_img[r][c] = clip(value)
    return conv_img

def get_mask(mask_type='prewitX'):
    '''
    Arg:
        mask_type - string
            prewitX
            prewitY
            embos1
            embos2
            laplace4
            laplace8
            unsharp4
            unsharp8        
    return :
        mask
            numpy 3 by 3 matrix
    '''
    if mask_type == 'prewitX':
        mask = [1, 1, 1, 0, 0, 0, -1, -1, -1]
    if mask_type == 'prewitY':
        mask = [1, 0, -1, 1, 0, -1, 1, 0, -1]
    if mask_type == 'embos1':
        mask = [1, 0, 0, 0, 0, 0, 0, 0, -1]
    if mask_type == 'embos2':
        mask = [0, 0, 1, 0, 0, 0, -1, 0, 0]
    if mask_type == 'laplace4':
        mask = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    if mask_type == 'laplace8':
        mask = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    if mask_type == 'unsharp4':
        mask = [0, -1, 0, -1, 5, -1, 0, -1, 0]
    if mask_type == 'unsharp8':
        mask = [-1, -1, -1, -1, 8, -1, -1, -1, -1]         
    mask = np.reshape(np.array(mask), [3, 3])
    return mask

def rgb2gray(rgbimage, option = 0):
    '''
        option 0
            Y = (Red + Green + Blue)/3
        option 1
            Y = Red * 0.2126 + Geeen * 0.7152 + Blue * 0.0722
        option 2
            Y = Red * 0.299 + Green * 0.587  + Blue * 0.114
    '''
    row, col, _= rgbimage.shape
    grayimage = np.zeros((row, col))
    option0 = [1/3, 1/3, 1/3]
    option1 = [0.2126, 0.7152, 0.0722]
    option2 = [0.299, 0.587, 0.144]
    weight = [option0, option1, option2]
    
    for i in range(row):
        for j in range(col):
            grayimage[i][j] = weight[option][0]*rgbimage[i][j][0] + weight[option][1]*rgbimage[i][j][1] + weight[option][2]*rgbimage[i][j][2]
            grayimage[i][j]*=255
            
    return grayimage

def flatten_img(img, option = False):
    row, col = img.shape
    image_size = row*col
    hist = np.zeros(256, dtype = int)
    cum_hist = np.zeros(256, dtype = int)
    norm_cum_hist = np.zeros(256)
    flat_image = np.zeros((row,col))
    hist2 = np.zeros(256)

    for i in range(row):
        for j in range(col):
            hist[int(img[i][j])]+=1

    cum_hist[0] = hist[0]
    for i in range(1, 256):
        cum_hist[i]=cum_hist[i-1]+hist[i]
    for i in range(256):
        norm_cum_hist[i] = float(cum_hist[i])/image_size

    for i in range(row):
        for j in range(col):
            flat_image[i][j] = norm_cum_hist[int(img[i][j])]*255
            hist2[int(flat_image[i][j])]+=1
    if option:
        return hist, cum_hist, norm_cum_hist, hist2, flat_image
    else :
        return flat_image