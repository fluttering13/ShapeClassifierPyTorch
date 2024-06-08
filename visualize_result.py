import pickle
import matplotlib.pyplot as plt
import os
from libs.utilty import *

def load_result(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_epoch_result(dataset_name,y_label_name):
    result_dict=load_result('./checkpoints/'+dataset_name+'.pkl')
    if not os.path.exists('./traing_result_pic'):
        os.mkdir('./traing_result_pic')
    if y_label_name=='training_loss_list':
        x_list=result_dict['epoch_list']
        y_list=result_dict['training_loss_list']
        plt.scatter(x_list,y_list)
        plt.title('epoch - train_loss_list')
        plt.xlabel('epoch')
        plt.ylabel(y_label_name)
        plt.savefig('./traing_result_pic/'+dataset_name+'_'+'epoch'+'_'+y_label_name+'.png')
        plt.close()

    elif y_label_name=='val_acc_list':
        x_list=result_dict['epoch_list'][::2]
        y_list=result_dict['val_acc_list']
        plt.scatter(x_list,y_list)
        plt.title('epoch - val_acc_list')
        plt.xlabel('epoch')
        plt.ylabel(y_label_name)
        plt.savefig('./traing_result_pic/'+dataset_name+'_'+'epoch'+'_'+y_label_name+'.png')
        plt.close()
    elif y_label_name=='validation_loss_list':
        x_list=result_dict['epoch_list'][::2]
        y_list=result_dict['validation_loss_list']
        plt.scatter(x_list,y_list)
        plt.title('epoch - val_loss_list')
        plt.xlabel('epoch')
        plt.ylabel(y_label_name)
        plt.savefig('./traing_result_pic/'+dataset_name+'_'+'epoch'+'_'+y_label_name+'.png')
        plt.close()
    pic_path='./traing_result_pic/'+dataset_name+'_'+'epoch'+'_'+y_label_name+'.png'
    return pic_path

data_set_name_list=['color_type_random_dataset','side_legnth_type_random_dataset','rotation_bool_dataset','hard_dataset']
y_label_name_list=['training_loss_list','validation_loss_list','val_acc_list']

for data_set_name in data_set_name_list:
    pic_path_list=[]
    for y_label_name in y_label_name_list:
        pic_path=plot_epoch_result(data_set_name,y_label_name)
        pic_path_list.append(pic_path)
    display_side_by_side(pic_path_list)