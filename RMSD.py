import cv2, os, sys
import matplotlib.pyplot as plt

def RMSD(questionID, target, master):
    print "questionID", questionID
    plt.subplot(121)
    plt.imshow(target, cmap='gray')
    plt.subplot(122)
    plt.imshow(master, cmap='gray')
    plt.show()
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        print "returning from 1"
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               print "returning from 2"
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                print "returning from 3"
                return -1
        else:
            print "returning from 4"
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

        return total_diff;

filetarget = sys.argv[2]
filemaster = sys.argv[3]
question = sys.argv[1]

target = cv2.imread(filetarget, 0)
master = cv2.imread(filemaster, 0)

print RMSD(int(question), target, master)
