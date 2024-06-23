import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
import os
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from scipy import spatial


PATH = '/home/raghosh/Documents/RS_Segmentation_Results/results_STEGO_unsupervised/Scratch_Training_Cls_4/results_STEGO_10_10_2022_EXP1_epoch_50/training_curve_arrSTEGO_10_10_2022.npy'

z = np.load(PATH)
print(np.shape(z))
print(np.min(z), np.max(z))

plt.plot(z)
plt.show()


class Dataset():
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path

        self.images = list()
        self.masks = list()
        self.preds = list()

        self.images_lst = list()
        self.masks_lst = list()

        for im in os.listdir(self.images_path):

            if 'pred_feature_map' in im:
                self.preds.append(os.path.join(self.images_path, im))

        self.preds = sorted(self.preds)
#        print(self.images)

        for msk in os.listdir(self.masks_path):

            if 'true_test_gt' in msk and '.npy' in msk:
                self.masks.append(os.path.join(self.masks_path, msk))

            if 'test_numpy_arr' in msk and '.npy' in msk:
                self.images.append(os.path.join(self.masks_path, msk))


        self.masks = sorted(self.masks)
        self.images = sorted(self.images)

    def get_test_image_mask_list(self):

        for im, msk in zip(self.preds, self.masks):

#            fig = plt.figure(figsize=(8, 8))

            x = np.load(im)
            print(x.shape)
            x = x.reshape(3, 320, 320)
            x = x.argmax(0)

            plt.imshow(x)
            plt.show()
            #x = x.argmax(0)
            #print(x.shape)

            #print(x.shape)

            #print(x.shape)

#            x = x.reshape(3, 320, 320)
#            x = x.argmax(0)
            #x = np.where(x==0, 2, x)

            #x = x + 4
            #x = np.where(x == 4, 2, x)
            #x = np.where(x == 7, 1, x)
            #x = np.where(x == 6, 1 ,x)
            #x = np.where(x == 5, 3, x)

            """ This combination is the best for results_STEGO_13_10_2022_cls_3_epoch_300 """
            x = x + 3
            #x = np.where(x == 3, 0, x)
            #x = np.where(x == 4, 2, x)

            #x = np.where(x == 5, 3, x)

            #x = np.where(x == 1, 64, x)
            #x = np.where(x == 2, 128, x)
            #x = np.where(x == 3, 256, x)

            #x = np.where(x == 1, 127, x)
            #x = x + 1
            #x = np.where(x == 1, 63, x)
            #x = np.where(x == 2, 127, x)
            #x = np.where(x == 3, 255, x)

            """ Here we would like to make the combination as per the requirements """
#            x = x + 3
#            x = np.where(x == 3, 1, x)
#            x = np.where(x == 5, 2, x)
#            x = np.where(x == 4, 3, x)

#            plt.imshow(x)
#            plt.show()

            y = np.load(msk)
            y = y[40:360, 40:360]
            y = y + 3
            y = np.where(y == 3, 1, y)
            y = np.where(y == 4, 0, y)
            y = np.where(y == 5, 2, y)
            y = np.where(y == 6, 3, y)
            #y = np.where(y == 0, 32, y)
            #y = np.where(y == 1, 64, y)
            #y = np.where(y == 2, 128, y)
            #y = np.where(y == 3, 256, y)
            #y = y - 1

            """ This operation is done just to compute the accuracy of the model """

            #x = np.where(y == 0, 0, x)

            rows, cols = 2, 2
            #result = 1 - spatial.distance.cosine(x.flatten(), y.flatten())
            fig = plt.figure(figsize=(8, 8))

            fig.add_subplot(2, 2, 1)
            plt.imshow(x)
            #plt.show()

            fig.add_subplot(2, 2, 2)
            plt.imshow(y)
            #plt.show()

            z = np.load(self.images[self.preds.index(im)])

            fig.add_subplot(2, 2, 3)
            plt.imshow(z[40:360, 40:360])

            plt.show()

            """ here, we are presenting the number of figures """


            self.images_lst.append(x)
            self.masks_lst.append(y)

        y_true = np.concatenate(self.images_lst, axis=1)
        y_pred = np.concatenate(self.masks_lst, axis=1)

        return [y_true, y_pred]



            #print(im)
#            print(result)

#            plt.imshow(y)
#            plt.show()

#            x1 = np.reshape(x[:1, 3:4, :, :], (400, 400))
#            plt.imshow(x1)
#            plt.show()
"""
            zero_arr= np.zeros((1, 4, 400, 400), dtype=float)
            one_arr = np.ones((1, 4, 400, 400), dtype=float)
            x_fmap = np.where(x >= 0.6, one_arr, zero_arr)

            h, w = np.shape(x)[2], np.shape(x)[3]

            zeros = np.zeros([h, w])

            label1 = np.where(x_fmap[:1, 0:1, :, :] == 1, 0, zeros)
            label2 = np.where(x_fmap[:1, 1:2, :, :] == 1, 1, zeros)
            label3 = np.where(x_fmap[:1, 2:3, :, :] == 1, 2, zeros)
            label4 = np.where(x_fmap[:1, 3:4, :, :] == 1, 3, zeros)

            final_gt = label1.reshape(h, w) + label2.reshape(h, w) + label3.reshape(h, w) + label4.reshape(h, w)

#            y = np.load(msk)

#            result = 1 - spatial.distance.cosine(final_gt.flatten(), y.flatten())

#            print(result)

#            plt.imshow(final_gt)
#            plt.axis('off')
#            plt.show()

            final_gt1 = np.where(y == 0, 0, final_gt)

            self.images_lst.append(final_gt1)
            self.masks_lst.append(y)

#        return [self.images_lst, self.masks_lst]

        y_true = np.concatenate(self.masks_lst, axis=1)

#        plt.imshow(np.concatenate([y_true[:, 8000:9200], y_true[:, 9600:10800]], axis=1), cmap='gray')
#        plt.show()

#        print(type(y_true), 'Type of True GT')
#        print(np.shape(y_true))

        #plt.imshow(y_pred[:, 2000:4000])
        #plt.show()
        y_pred = np.concatenate(self.images_lst, axis=1)

#        plt.imshow(np.concatenate([y_pred[:, 8000:9200], y_pred[:, 9600:10800]], axis=1))
#        plt.show()


        return [y_true, y_pred]

#        return [self.images_lst, self.masks_lst]
"""

#fig = plt.figure(figsize=(10, 7))

            #fig.add_subplot(rows, cols, 1)

# showing image
            #plt.imshow(final_gt)
            #plt.axis('off')
            #plt.title("First")

# Adds a subplot at the 2nd position
            #fig.add_subplot(rows, cols, 2)

# showing image
            #plt.imshow(y)
            #plt.axis('off')
            #plt.title("Second")

#            f, axarr = plt.subplots(1, 2)
#            axarr[0,0].imshow(final_gt)
#            axarr[0,1].imshow(y)
            #fig.add_subplot(rows, cols, 3)

            #plt.imshow(final_gt1)
            #plt.axis('off')
            #plt.title("Third")

            #plt.show()


            #y = np.load(msk)

        #for msk in self.masks:

            #x = np.load(msk)
            #print(x.shape)
            #plt.imshow(x)
            #plt.show()
            #x = x[:, :, 1]
            #x = np.reshape(x, (400, 400))
            #plt.imshow(x)
            #plt.show()

#class Confusion_Matrix:
    #def __init__(self, y_true, y_pred):
#       self.y_true = y_true
#       self.y_pred = y_pred


#results_18_08_2021_TransUNET_FFT_EXP1


#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'results_Attn_UNet_20_12_2021')

""" The best results in terms of the accuracy of the unsupervised semantic segmentation method """

#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'results_STEGO_unsupervised',
#                           'results_STEGO_13_10_2022_cls_3_epoch_300',
#                           )


#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'results_STEGO_unsupervised',
#                           'Scratch_Training_Cls_3',
#                           'feats_192_cls3',
#                           'results_STEGO_01_12_2022_cls_3_epoch_300_SpatialConsistencyUpsample_h_w_320x320_with_cluster_loss_preds'
#                           )

images_path = os.path.join(os.path.expanduser('~'),
                           'Documents',
                           'RS_Segmentation_Results',
                           'results_STEGO_unsupervised',
                           'Scratch_Training_Cls_3',
                           'feats_192_cls3',
                           'results_ESTEGO_22_02_2023_cls_3_epoch_100_bs_16_SpatialConsistencyUpsample_h_w_320x320_pos_intra_08_pos_inter_02_data_1200_iteration19'
                           #'Sudipan_feats_128',
                           #'experiment_13_10_2023_cls_3',
                           #'results_STEGO_16_02_2023_cls_3_epoch_100_SpatialConsistencyUpsample_h_w_320x320_with_cluster_probe_loss_CRF_refinement_results_posintra_shift_23_pos_intershift_17'
                           )

#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'results_STEGO_unsupervised',
#                           'Scratch_Training_Cls_3',
#                           'SpatialConsistencyLoss_results_STEGO_30_11_2022_epoch_300_cluster_probe',
#                           )

#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'results_20_05_2021_Hybrid_TrnsUNET_TransFUSE')


masks_path = os.path.join(os.path.expanduser('~'),
                          'Documents',
                          'RS_Segmentation_Results',
                          'mask_arr_TransUNET_400x400')


#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'TransSounder_radargram_qualifying'
#                           )

#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'TransUNet_Qualifying_rsde_juice_06_10_2021'
#                           )

#images_path = os.path.join(os.path.expanduser('~'),
#                           'Documents',
#                           'RS_Segmentation_Results',
#                           'TransFUSE_radargram_Qualifying'
#                           )

#masks_path = os.path.join(os.path.expanduser('~'),
#                          'Documents',
#                          'RS_Segmentation_Results',
#                          'TransUNet_radargram_Qualifying')


x = Dataset(images_path, masks_path)
#y = x.get_similarity()
y = x.get_test_image_mask_list()


def get_confusion_matrix(true, predicted):

    true_list = list(itertools.chain.from_iterable(true.tolist()))
    print(true_list[:10])
    predicted_list = list(itertools.chain.from_iterable(predicted.tolist()))
    print(predicted_list[:10])
    actu = pd.Series(true_list, name='Actual')
    pred = pd.Series(predicted_list, name='Predicted')

    x = pd.crosstab(actu, pred)
    plt.imshow(x)
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.6f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


y_true = list(itertools.chain.from_iterable(y[0].tolist()))
y_pred = list(itertools.chain.from_iterable(y[1].tolist()))

cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
#np.set_printoptions(precision=6)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3], title='Confusion matrix, without normalization')

print(precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average='macro'))

print(precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 'precision')

print(recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 'recall')

print(f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 'f1score')

print(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3]), 'kappa score')

print(jaccard_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 'jaccard_score')

print(accuracy_score(y_true, y_pred), 'accuracy_score')
