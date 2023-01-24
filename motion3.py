import tensorflow as tf
from core_warp import dense_image_warp
from core_costvol import cost_volume

pyr_lvls = 6
flow_pred_lvl = 2
dbg = False
use_dense_cx = True
use_res_cx = True
search_range = 4

def extract_features(x_tnsr, name='featpyr'):
    assert(1 <= pyr_lvls <= 6)
    if dbg:
        print(f"Building feature pyramids (c11,c21) ... (c1{pyr_lvls},c2{pyr_lvls})")
    # Make the feature pyramids 1-based for better readability down the line
    num_chann = [None, 16, 32, 64, 96, 128, 196]
    c1, c2 = [None], [None]
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
            for lvl in range(1, pyr_lvls + 1):
                # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                # reuse is set to True because we want to learn a single set of weights for the pyramid
                # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                f = num_chann[lvl]
                x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name=f'conv{lvl}a', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}aa', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}aa')
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}b', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1, name=f'{name}{lvl}')
                pyr.append(x)
    return c1, c2

def corr_f(c1, warp, lvl, name='corr'):
    op_name = f'corr{lvl}'
    if dbg:
        print(f'Adding {op_name} with inputs {c1.op.name} and {warp.op.name}')
    with tf.name_scope(name):
        return cost_volume(c1, warp, search_range, op_name)

def predict_flow(corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
    op_name = f'flow{lvl}'
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        if c1 is None and up_flow is None and up_feat is None:
            if dbg:
                print(f'Adding {op_name} with input {corr.op.name}')
            x = corr
        else:
            if dbg:
                msg = f'Adding {op_name} with inputs {corr.op.name}, {c1.op.name}, {up_flow.op.name}, {up_feat.op.name}'
                print(msg)
            x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_0')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_1')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_2')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_3')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_4')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # will also be used as an input by the context network
        upfeat = tf.concat([act, x], axis=3, name=f'upfeat{lvl}') if use_dense_cx else act

        flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

        return upfeat, flow


def warp_f(c2, sc_up_flow, lvl, name='warp'):
    op_name = f'{name}{lvl}'
    if dbg:
        msg = f'Adding {op_name} with inputs {c2.op.name} and {sc_up_flow.op.name}'
        print(msg)
    with tf.name_scope(name):
        return dense_image_warp(c2, sc_up_flow, name=op_name)

def refine_flow(feat, flow, lvl, name='ctxt'):
    op_name = f'refined_flow{lvl}'
    if dbg:
        print(f'Adding {op_name} sum of dc_convs_chain({feat.op.name}) with {flow.op.name}')
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}1')
        x = tf.nn.leaky_relu(x, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name=f'dc_conv{lvl}2')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name=f'dc_conv{lvl}3')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name=f'dc_conv{lvl}4')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name=f'dc_conv{lvl}5')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}6')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}7')

        return tf.add(flow, x, name=op_name)

def deconv(x, lvl, name='up_flow'):
    op_name = f'{name}{lvl}'
    if dbg:
        print(f'Adding {op_name} with input {x.op.name}')
    with tf.variable_scope('upsample'):
        # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
        return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)

def nn(x_tnsr, name='pwcnet'):
   with tf.variable_scope(name):
        # Extract pyramids of CNN features from both input images (1-based lists))
        c1, c2 = extract_features(x_tnsr)

        flow_pyr = []


        for lvl in range(pyr_lvls, flow_pred_lvl - 1, -1):

            if lvl == pyr_lvls:
                # Compute the cost volume
                corr = corr_f(c1[lvl], c2[lvl], lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr, None, None, None, lvl)
            else:
                # Warp level of Image1's using the upsampled flow
                scaler = 20. / 2**lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                warp = warp_f(c2[lvl], up_flow * scaler, lvl)

                # Compute the cost volume
                corr = corr_f(c1[lvl], warp, lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr, c1[lvl], up_flow, up_feat, lvl)

            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

            if lvl != flow_pred_lvl:
                if use_res_cx:
                    flow = refine_flow(upfeat, flow, lvl)

                 # Upsample predicted flow and the features used to compute predicted flow
                flow_pyr.append(flow)

                up_flow = deconv(flow, lvl, 'up_flow')
                up_feat = deconv(upfeat, lvl, 'up_feat')
            else:
                # Refine the final predicted flow
                flow = refine_flow(upfeat, flow, lvl)
                flow_pyr.append(flow)

                # Upsample the predicted flow (final output) to match the size of the images
                scaler = 2**flow_pred_lvl
                if dbg:
                    print(f'Upsampling {flow.op.name} by {scaler} in each dimension.')
                size = (lvl_height * scaler, lvl_width * scaler)
                flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scaler
                break

   return flow_pred, flow_pyr
