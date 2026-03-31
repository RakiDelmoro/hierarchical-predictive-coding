import torch


def ipc_update_weights(model, z1, z2, z1_pred, z2_pred, x_tp1, x_pred, lr=1e-3):
    e_x = x_tp1 - x_pred

    dec_loss = (e_x * z1_pred).mean()
    model.decoder.deconv[-1].weight.grad = None
    model.decoder.deconv[-1].weight.grad = -dec_loss * lr

    e_z1 = z1_pred - model.z1_predict(model.up_z2_to_z1(z2_pred)).detach()
    z1_loss = e_z1.mean()
    model.z1_predict[-1].weight.grad = None
    model.z1_predict[-1].weight.grad = -z1_loss * lr

    e_z2 = z2_pred - model.z2_predict(model.z2_transition(z2, z1_pred)).detach()
    z2_loss = e_z2.mean()
    model.z2_predict[-1].weight.grad = None
    model.z2_predict[-1].weight.grad = -z2_loss * lr

    z1_trans_loss = (z1_pred - model.z1_predict(model.up_z2_to_z1(z2))).mean()
    model.up_z2_to_z1.weight.grad = None
    model.up_z2_to_z1.weight.grad = -z1_trans_loss * lr

    z2_trans_loss = (z2_pred - model.z2_transition(z2, z1_pred)).mean()
    model.z2_transition.gru.conv_q.weight.grad = None
    model.z2_transition.gru.conv_q.weight.grad = -z2_trans_loss * lr

    return dec_loss.item(), z1_loss.item(), z2_loss.item()
