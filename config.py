# workspace = ""
workspace = "/vol/vssp/msos/qk/workspaces/ICASSP2018_dcase"

# config
sample_rate = 16000.
n_window = 1024
n_overlap = 360      # ensure 240 frames in 10 seconds
max_len = 240        # sequence max length is 10 s, 240 frames. 
step_time_in_sec = float(n_window - n_overlap) / sample_rate

# Id of classes
ids = ['/m/0284vy3', '/m/05x_td', '/m/02mfyn', '/m/02rhddq', '/m/0199g', 
       '/m/06_fw', '/m/012n7d', '/m/012ndj', '/m/0dgbq', '/m/04qvtq', 
       '/m/03qc9zr', '/m/0k4j', '/t/dd00134', '/m/01bjv', '/m/07r04', 
       '/m/04_sv', '/m/07jdr']

# Name of classes
lbs = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps', 
       'Bicycle', 'Skateboard', 'Ambulance (siren)', 
       'Fire engine, fire truck (siren)', 'Civil defense siren', 
       'Police car (siren)', 'Screaming', 'Car', 'Car passing by', 'Bus', 
       'Truck', 'Motorcycle', 'Train']
          
idx_to_id = {index: id for index, id in enumerate(ids)}
id_to_idx = {id: index for index, id in enumerate(ids)}
idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)