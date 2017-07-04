import numpy as np

def k_means(gray_img, clusters = 2):
    '''
    Apply k means clustering on gray_image
    if cluster == 2 => binarize

    Arg :
        gray_img : 2D image [height, width]

        clusters : int
            number of clusters in k-means
    Return :
        new_img : 2D image [height, width]

    '''
    height, width = gray_img.shape
    histogram = get_histogram(gray_img)
    center = np.linspace(0,255, clusters)
    sum_ = np.zeros(clusters)
    count_ = np.zeros(clusters)

    for i in range(256):
        min_index = np.argmin(np.abs(i-center))
        count_[min_index]+=histogram[i]
        sum_[min_index]+=i*histogram[i]

    new_center = np.zeros(clusters)
    for i in range(clusters):
        new_center[i] = 0 if count_[i]==0 else sum_[i]/count_[i] 
        
    while np.max(np.abs(new_center-center)) > 1:
        for i in range(clusters):
            center[i] = new_center[i]
        sum_ = np.zeros(clusters)
        count_ = np.zeros(clusters)
        for i in range(256):
            min_index = np.argmin(np.abs(i-center))
            count_[min_index]+=histogram[i]
            sum_[min_index]+=i*histogram[i]

        new_center = np.zeros(clusters)
        for i in range(clusters):
            new_center[i] = 0 if count_[i]==0 else sum_[i]/count_[i] 

    for i in range(clusters):
        center[i] = new_center[i]/255

    new_img = np.zeros((height, width))
    new_color = np.linspace(0, 1, clusters)

    for h in range(height):
        for w in range(width):
            min_index = np.argmin(np.abs(gray_img[h][w]-center))
            new_img[h][w] = new_color[min_index]

    return new_img

def freq_filter(img, freq = 10.0):
    '''
    Arg :
        img - 2D array [height, width]
        freq - float
    return :
        low pass filtered image, high pass filtered image - 2D array [height, width]
        high pass filtered image is normalized to be [0, 1]
    '''
    fft_imag = np.fft.fft2(img)
    height, width = fft_imag.shape
    low_pass = np.zeros((height, width), dtype = 'complex128')
    high_pass = np.zeros((height, width), dtype = 'complex128')
    for h in range(height):
        low_pass[h][0] = fft_imag[h][0]
    for w in range(width):
        low_pass[0][w] = fft_imag[0][w]

    for h in range(height):
        for w in range(width):
            dist_x = min(abs(h), abs(height-h))
            dist_y = min(abs(w), abs(width-w))
            dist = dist_x*dist_x+dist_y*dist_y
            low_pass[h][w] = fft_imag[h][w]*np.exp(-dist/freq/freq)
            high_pass[h][w] = fft_imag[h][w]-low_pass[h][w]
    
    inverse_low_pass = np.fft.ifft2(low_pass).real
    inverse_high_pass = np.fft.ifft2(high_pass).real
    clipped_inverse_high_pass = clip_2d(inverse_high_pass-np.min(inverse_high_pass))
    return inverse_low_pass, clipped_inverse_high_pass

def contrast(img, degree = 0.3):
    '''
    Arg:
        img : 2D array [0,1] 
            gray image basically dark image
        degree : float 
            used to delete main dark image before flatten
    return : 
        contrasted image dark image deleted
    '''
    height, width = img.shape
    std = np.std(img)
    for h in range(height):
        for w in range(width):
            if std*degree>img[h][w]:
                img[h][w] = 0
    return img

def HSV2img(H_img, S_img, V_img):
    '''
    Arg :
        H_img - 2D array [height, width]
            each element [0, 360]
        S_img - 2D array [height, width]
            each element [0, 1]
        V_img - 2D array [height, width]
            each element [0, 1]
    return :
        reconst_image - 3D array [height, width, 3] 
    '''
    height, width = H_img.shape
    reconst_img = np.zeros((height, width, 3))
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
            reconst_img[h][w][0] = r
            reconst_img[h][w][1] = g
            reconst_img[h][w][2] = b
    return reconst_img

def get_HSV(img):
    '''
    Arg:
        img - 3D np array [height, weight, channel(=3)]
            [0, 1]
    return:
        H_img - 2D array [height, width]
            []
        S_img - 2D array [height, width]
            [0, 1]
        V_img - 2D array [height, width]
            [0, 1]
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
            [0, 1]
    return:
        H_img - 2D array [height, width]
            []
        S_img - 2D array [height, width]
            [0, 1]
        I_img - 2D array [height, width]
            [0, 1]
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
            [0, 1]
    return:
        H_img - 2D array [height, width]
            []
        S_img - 2D array [height, width]
            [0, 1]
        L_img - 2D array [height, width]
            [0, 1]
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


def clip(x, vmax = 1.0, vmin = 0.0):
    '''
    Arg:
        x - float
        vmax - float
            default to be 1.0
        vmin - float
            default to be 0.0
    return :
        clipped x [vmin, vmax]
    '''
    if x>vmax:
        return vmax
    if x<vmin:
        return vmin
    return x

def clip_2d(array, vmax = 1, vmin = 0 ):
    '''
    Arg:
        array - 2D array
        vmax - float
            default to be 1.0
        vmin - float
            default to be 0.0
    return :
        clipped array [vmin, vmax]
    '''
    height, width = array.shape
    clipped_array = np.zeros_like(array)
    for h in range(height):
        for w in range(width):
            clipped_array[h][w] = clip(array[h][w], vmax=vmax, vmin=vmin)
    return clipped_array

def cycle_clip(value, turn = 360):
    '''
    Default to be used as angle converter
    ex)
        -10 -> 350
        370 -> 10
    
    Arg :
        value - float 
        turn - int
    return :
        value between [0, turn]
    '''
    return value-turn*(np.floor(value/turn))

def cycle_clip_2d(array, turn= 360):
    '''
    Default to be used as angle converter for 2D array

    Arg :
        array - 2D array float 
        turn - int
    return :
        array with every value between [0, turn]
    '''
    height, width = array.shape
    clipped_array = np.zeros_like(array)
    for h in range(height):
        for w in range(width):
            clipped_array[h][w] = cycle_clip(array[h][w],turn)
    return clipped_array

def conv(img, filt):
    '''
    Arg :
        img - 2D array
        filt - 2D array
    return :
        image convoluted with filt
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

def rgb2gray(rgbimg, option = 0):
    '''
    Arg :
        rgbimg - 3D array [height, weight, 3]
        option - int 0, 1, 2 
            option 0
                Y = (Red + Green + Blue)/3
            option 1
                Y = Red * 0.2126 + Geeen * 0.7152 + Blue * 0.0722
            option 2
                Y = Red * 0.299 + Green * 0.587  + Blue * 0.114
    Return :
        grayimg - 2D array [height, weight] 
    '''
    row, col, _= rgbimg.shape
    grayimg = np.zeros((row, col))
    option0 = [1/3, 1/3, 1/3]
    option1 = [0.2126, 0.7152, 0.0722]
    option2 = [0.299, 0.587, 0.144]
    weight = [option0, option1, option2]
    
    for i in range(row):
        for j in range(col):
            grayimg[i][j] = weight[option][0]*rgbimg[i][j][0] + weight[option][1]*rgbimg[i][j][1] + weight[option][2]*rgbimg[i][j][2]
            
    return grayimg

def get_histogram(gr_img):
    row, col = gr_img.shape
    hist = np.zeros(256, dtype =int)
    for i in range(row):
        for j in range(col):
            hist[int(gr_img[i][j]*255)]+=1
    return hist

def flatten_img(gr_img, option = False):
    '''
    Arg :
        gr_img - gray image [height, width]
    result : 
        flat-image - flatten with histogram [height, weight]
    '''
    row, col = gr_img.shape
    image_size = row*col

    cum_hist = np.zeros(256, dtype = int)
    norm_cum_hist = np.zeros(256)
    flat_image = np.zeros((row,col))
    hist = get_histogram(gr_img)

    cum_hist[0] = hist[0]
    
    for i in range(1, 256):
        cum_hist[i]=cum_hist[i-1]+hist[i]

    for i in range(256):
        norm_cum_hist[i] = float(cum_hist[i])/image_size

    for i in range(row):
        for j in range(col):
            flat_image[i][j] = norm_cum_hist[int(gr_img[i][j]*255)]

    hist2 = get_histogram(flat_image)

    if option:
        return hist, cum_hist, norm_cum_hist, hist2, flat_image
    else :
        return flat_image