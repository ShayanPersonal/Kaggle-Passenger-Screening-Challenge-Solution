import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import shutil
from skimage import io
import gpustat
import psutil

def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 5)
    return h

def read_data(infile, scale=True):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
        if scale:
            data = data * h['data_scale_factor'] #scaling factor
        else:
            data = (data, h['data_scale_factor'])
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag

def anim_to_images_all(image_count):
    for filename in os.listdir('a3daps'):
        anim_to_images(filename.split('.')[0], image_count)

def anim_to_images(name, image_count = 4):
    os.makedirs("images/{}".format(name))
    for i, image in enumerate(get_x_views("a3daps/" + name + ".a3daps", image_count)):
        io.imsave("images/{}/{}_{}.png".format(name, name, i), image)

def plot_image(path):
    matplotlib.rc('animation', html='html5')
    data = read_data(path)
    fig = plt.figure(figsize = (16,16))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=300, blit=True)

def get_x_views(filename, x=4):
    data = read_data(filename)
    # Upright the image
    views = data.shape[2]
    return [np.flipud(data[:, :, i].transpose()) for i in range(0, views, views // x)]

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def show_views(filename, x=16):
    for im in get_x_views(filename, x=x):
        plt.imshow(im[::1, ::1], cmap = 'viridis')
        plt.show()

def find_test_data():
    names = [filename.split('.')[0] for filename in os.listdir('aps/')]
    train_file = open('stage1_labels.csv')
    train_file.readline()
    train_data = set()
    for line in train_file:
        train_data.add(line.strip().split(',')[0].split('_')[0])
    for name in names:
        if name not in train_data:
            print("yo")
            shutil.move("aps/{}.aps".format(name), 'test/')

def view_dataset():
    labels = open("stage1_labels.csv")
    labels.readline()
    old_name = None
    for line in labels:
        name, _ = line.split('_')
        if name != old_name:
            print('-----------------------')
            if old_name:
                plt.show(plot_image('aps/{}.aps'.format(old_name)))
            old_name = name
        print(line.strip())

def read_blanks():
    labels = open("blanks.txt")
    for i, name in enumerate(labels):
        name = name.strip()
        plt.imshow("people/{}.png".format(i), np.flipud(read_data('aps/{}.aps'.format(name))[:, :, 0].transpose()))
        plt.show()

def show_test():
    for (dirpath, dirnames, filenames) in os.walk("test/"):
        for i, f in enumerate(filenames):
            name, ext = f.split('.')
            if ext == "a3daps":
                continue
            plt.imsave("test_ims/{}.png".format(i), np.flipud(read_data('test/{}.aps'.format(name))[:, :, 0].transpose()))
    
def view_dataset_x(views=16):
    labels = open("stage1_labels.csv")
    labels.readline()
    old_name = None
    for line in labels:
        name, _ = line.split('_')
        if name != old_name:
            print('-----------------------')
            if old_name:
                show_views('aps/' + old_name + '.aps', views)
            old_name = name
        print(line.strip())

def analyse_predictions(filename):
    csv_in = open(filename)
    csv_in.readline()
    avgs = [[0,0,0] for _ in range(17)]          # avg, min, max
    for _ in range(1700):
        x, y = csv_in.readline().split(',')
        _, zone = x.split('_')
        zone = int(zone[4:])
        avgs[zone-1][0] += float(y) / 100
        avgs[zone-1][1] = min(avgs[zone-1][1], float(y))
        avgs[zone-1][2] = max(avgs[zone-1][2], float(y))
    for i, val in enumerate(avgs):
        print(str(i+1) + " " + str(val))

def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]/2.**30
    print("GPU: {}/{}".format(item["memory.used"], item["memory.total"]))
    print("CPU: {0:.2f}/32".format(memory_use))


def super_image():
    labels = open("stage1_labels.csv")
    labels.readline()
    old_name = None
    super_image = None
    i = 0
    for line in labels:
        name, _ = line.split('_')
        if name != old_name:
            if old_name:
                i += 1
                im = read_data('aps/{}'.format(old_name + ".aps"))
                print(name)
                if super_image is not None:
                    super_image = np.maximum(super_image, im)
                else:
                    super_image = im
            old_name = name
    print(i)
    print(super_image.shape)
    print(super_image.max())
    print(super_image.min())
    for i in range(16):
        plt.imshow(super_image[:, :, i])
        plt.show()

def move_back_test():
    for filename in os.listdir('test/'):
        name, ext = filename.split('.')
        if ext == "a3daps":
            shutil.move("test/{}".format(filename), 'a3daps/')
        if ext == "aps":
            shutil.move("test/{}".format(filename), 'aps/')


if __name__ == "__main__":
    analyse_predictions("predictions/13318918/predictions_13318918_0.csv")
    #view_dataset()
    #show_views("aps/42181583618ce4bbfbc0c4c300108bf5.aps", 16)
    #plt.show(plot_image("test/f22232983210eea304a0ad9cbe807d27.aps"))
    #view_dataset()
    #super_image()
    #read_blanks()
    #show_test()
    pass