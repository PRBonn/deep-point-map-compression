
    
def remove_sequential(network):
    all_layers = []
    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, nn.Sequential):
            all_layers += remove_sequential(layer)
        else:
            all_layers += [layer]

    return all_layers
    
    
    
    def visualizeSampling(self):
        t_gpu = time.time()
        print(20*'-')
        print('Visualize Deformation')
        self.loadModel(best=False)
        # torch.backends.cudnn.enabled = False
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)

        print(f'Model gpu ({time.time()-t_gpu}s)')

        t_iter = time.time()
        input_dict = self.submaps.train_iter_ordered.next()
        print(f'Got train iter ({time.time()-t_iter}s)')
        input_dict['points'] = input_dict['points'].cuda()
        input_dict['features'] = input_dict['features'].cuda()
        input_dict['norm_range'] = input_dict['norm_range'].to(
            self.device)
        print(20*'#')

        all_layers = remove_sequential(self.encoder_model)
        all_layers += remove_sequential(self.decoder_model)
        # print(all_layers)
        pcl = []
        clr = []
        print(f'Time until start ({time.time()-t_gpu}s)')

        for m in all_layers:
            print(type(m))
            in_pcl = input_dict['points'].detach().cpu().numpy()
            input_dict = m(input_dict)
            out_pcl = input_dict['points'].detach().cpu().numpy()

            if(type(m) in [network.SampleKPConvBlock, network.InterKPConvBlock, network.RandomSampleKPConv, network.ResnetConv, network.FPSResnetConv, network.FPSSamplingConv, network.GridSampleConv]):
                print(input_dict['points'].shape, input_dict['features'].shape)
                print('kernel_radius', m.kernel_radius)
                pts, clrs = pcu.colorizeConv(
                    in_pcl, out_pcl, m.kernel_radius, m.max_nr_neighbors, kernel_points=m.kp_conv.kernel_points.cpu())
                pcl.append(pts)
                clr.append(clrs)
            if(type(m) in [network.LinearDeconv, network.AdaptiveDeconv]):
                print(input_dict['points'].shape, input_dict['features'].shape)
                print('kernel_radius', m.kernel_radius)
                pts, clrs = pcu.colorizeConv(
                    in_pcl, out_pcl, m.kernel_radius, max_nr_neighbors=m.upsampling_rate)
                pcl.append(pts)
                clr.append(clrs)
        pcu.visualizeConv(pcl, clr)

    def visualizeDeformation(self):
        print(20*'-')
        print('Visualize Deformation')
        self.loadModel(best=True)
        # torch.backends.cudnn.enabled = False
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)

        input_dict = self.submaps.train_iter_ordered.next()
        input_dict['points'] = input_dict['points'].cuda()
        input_dict['norm_range'] = input_dict['norm_range'].to(
            self.device)


        all_layers = remove_sequential(self.encoder_model)
        all_layers += remove_sequential(self.decoder_model)
        # print(all_layers)
        pcl = []
        clr = []
        for m in all_layers:
            print(type(m))
            in_pcl = input_dict['points'].detach().cpu().numpy()
            input_dict = m(input_dict)
            out_pcl = input_dict['points'].detach().cpu().numpy()
            features = input_dict['features'].cpu()

            if isinstance(m, network.OriginalKpConv) and m.kp_conv.deformable:
                print(f'features min {features.min()}, max {features.max()}')
                print(input_dict['points'].shape, input_dict['features'].shape)
                print('kernel_radius', m.kernel_radius)

                print('deformed kp:', m.kp_conv.deformed_KP.shape)
                pts, clrs = pcu.colorizeConv(
                    in_pcl, out_pcl, m.kernel_radius, m.max_nr_neighbors, kernel_pos=m.kp_conv.deformed_KP.detach().cpu().numpy(), kernel_points=m.kp_conv.kernel_points.detach().cpu().numpy())
                pcl.append(pts)
                clr.append(clrs)
        pcu.visualizeConv(pcl, clr)