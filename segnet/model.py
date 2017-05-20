from common.layers import convs, max_pool_with_mask, unpool, conv2d, max_pool, unpool_cpu

def inference(images, phase):
    # the encoder structure as vgg16
    conv1_1 = convs('conv1_1', images, 3, 64, phase)
    conv1_2 = convs('conv1_2', conv1_1, 64, 64, phase)
    pool1 = max_pool(conv1_2, 2, 2)
    #pool1, mask1 = max_pool_with_mask(conv1_2, 2, 2)

    conv2_1 = convs('conv2_1', pool1, 64, 128, phase)
    conv2_2 = convs('conv2_2', conv2_1, 128, 128, phase)
    pool2 = max_pool(conv2_2, 2, 2)
    #pool2, mask2 = max_pool_with_mask(conv2_2, 2, 2)

    conv3_1 = convs('conv3_1', pool2, 128, 256, phase)
    conv3_2 = convs('conv3_2', conv3_1, 256, 256, phase)
    conv3_3 = convs('conv3_3', conv3_2, 256, 256, phase)
    pool3 = max_pool(conv3_3, 2, 2)
    #pool3, mask3 = max_pool_with_mask(conv3_3, 2, 2)

    conv4_1 = convs('conv4_1', pool3, 256, 512, phase)
    conv4_2 = convs('conv4_2', conv4_1, 512, 512, phase)
    conv4_3 = convs('conv4_3', conv4_2, 512, 512, phase)
    pool4 = max_pool(conv4_3, 2, 2)
    #pool4, mask4 = max_pool_with_mask(conv4_3, 2, 2)

    conv5_1 = convs('conv5_1', pool4, 512, 512, phase)
    conv5_2 = convs('conv5_2', conv5_1, 512, 512, phase)
    conv5_3 = convs('conv5_3', conv5_2, 512, 512, phase)
    pool5 = max_pool(conv5_3, 2, 2)
    #pool5, mask5 = max_pool_with_mask(conv5_3, 2, 2)

    # the decoder part, unpool and conv
    # up5 = unpool(pool5, mask5)
    up5 = unpool_cpu(pool5)
    up5c = convs('up5c', up5, 512, 512, phase)
    up5b = convs('up5b', up5c, 512, 512, phase)
    up5a = convs('up5a', up5b, 512, 512, phase)

    #up4 = unpool(up5a, mask4)
    up4 = unpool_cpu(up5a)
    up4c = convs('up4c', up4, 512, 512, phase)
    up4b = convs('up4b', up4c, 512, 512, phase)
    up4a = convs('up4a', up4b, 512, 256, phase)

    #up3 = unpool(up4a, mask3)
    up3 = unpool_cpu(up4a)
    up3c = convs('up3c', up3, 256, 256, phase)
    up3b = convs('up3b', up3c, 256, 256, phase)
    up3a = convs('up3a', up3b, 256, 128, phase)

    #up2 = unpool(up3a, mask2)
    up2 = unpool_cpu(up3a)
    up2b = convs('up2b', up2, 128, 128, phase)
    up2a = convs('up2a', up2b, 128, 64, phase)

    #up1 = unpool(up2a, mask1)
    up1 = unpool_cpu(up2a)
    up1b = convs('up1', up1, 64, 64, phase)
    out = conv2d(name='out', inputs=up1b, input_channels=64,
                 output_channels=3, kernel=3, stride=1)
    return out

