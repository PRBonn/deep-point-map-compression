import torch 

# import depoco.sample_net_trainer as snt
import depoco.architectures.original_kp_blocks as okp
import chamfer3D.dist_chamfer_3D
import depoco.architectures.network_blocks as network_blocks

def linDeconvRegularizer(net, weight,gt_points):
    cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    loss = torch.tensor(
            0.0, dtype=torch.float32, device=gt_points.device)  # init loss
    for m in net.modules():
        if (isinstance(m, network_blocks.LinearDeconv) or isinstance(m,network_blocks.AdaptiveDeconv)):
            d_map2transf, d_transf2map, idx3, idx4 = cham_loss(
                    gt_points.unsqueeze(0), m.points.unsqueeze(0))
            loss += (0.5 * d_map2transf.mean() +
                    0.5 * d_transf2map.mean())
    return weight * loss

# From KPCONV
def p2p_fitting_regularizer(net,deform_fitting_power=1.0,repulse_extent=1.2):
    l1 = torch.nn.L1Loss()
    fitting_loss = 0
    repulsive_loss = 0
    # print(20*'-')
    for m in net.modules():
        if isinstance(m, okp.KPConv) and m.deformable:
            # print(m)

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(m.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += l1(rep_loss, torch.zeros_like(rep_loss)) / m.K

    return deform_fitting_power * (2 * fitting_loss + repulsive_loss)
