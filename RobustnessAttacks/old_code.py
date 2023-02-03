



def get_2nd_stage_inputs(segmentation_model, data, transform_stage2, args):
    # inputs, labels, attr_labels, segmentation, bounding_box = data
    inputs, labels, attr_labels, _, _ = data
    inputs = inputs.float().cuda()

    segmentation_model.eval()
    segmentation_outputs = segmentation_model(inputs)  # output is list of #b dicts

    stage2_inputs = []
    attr_labels_remaining = []
    for i in range(args.batch_size):
        # scores = outputs[i]['scores']  # TODO ggf. threshold nutzen!

        # get the masks TODO unused for now
        # masks = (outputs[i]['masks'] > 0.5).squeeze().detach()

        bboxes = segmentation_outputs[i]['boxes'].detach()
        label_names = [coco_names[x] for x in segmentation_outputs[i]['labels']]

        if 'bird' in label_names:
            areas = torch.stack([bb for label, bb in zip(label_names, bboxes) if label == 'bird']).type(torch.IntTensor)
            # TODO tensor -> numpy array
            # crop image around bounding box of bird
            if areas.shape[0] > 1:
                x1y1 = torch.amin(areas[:, :2], axis=0)  # min x1, y1 # TODO min auf jede spalte aufrufen?
                x2y2 = torch.amax(areas[:, 2:], axis=0)  # max x2, y2
                # bounding box [x1, y1, x2, y2] format
                bounding_box = torch.hstack([x1y1, x2y2])
                stage2_inputs.append(inputs[i, :, bounding_box[0].item():bounding_box[2].item() + 1,
                                     bounding_box[1].item():bounding_box[3].item() + 1])
                attr_labels_remaining.append(attr_labels[i])
                # TODO numpy array -> tensor
            elif areas.shape[0] == 1:
                areas = areas[0]
                stage2_inputs.append(
                    inputs[i, :, areas[0].item():areas[2].item() + 1, areas[1].item():areas[3].item() + 1])
                attr_labels_remaining.append(attr_labels[i])
            else:
                # print('find some way to handle non-bird segmentations!')  # TODO
                pass

    # fit each tensor to the same size (224x224 for resnet34) and make a batch out of it again
    stage2_inputs = torch.stack([transform_stage2(x) for x in stage2_inputs])
    attr_labels_remaining = torch.stack(attr_labels_remaining)
    #print('attr_labels.shape: ', attr_labels_remaining.shape)
    return stage2_inputs.cuda(), attr_labels_remaining.cuda()


def attack_XtoC(concept_model_dir, end_model_dir, num_attr=3, eps=0.25):
    use_attr = True
    feature_group_results = False

    # setup model_templates (model 1 until bottleneck or baseline model) (x->c or no bottleneck)
    model = torch.load(concept_model_dir)
    model_end = torch.load(end_model_dir)

    model.use_relu = False
    model.use_sigmoid = False
    model.cy_fc = None
    #model.eval()
    #model_end.eval()

    full_model = ModelFullModel(concept_model=model, end_model=model_end, use_sigmoid=True, return_attributes=True)
    full_model.eval()

    # plug it together to a new model
    """new_model = torch.nn.Sequential(
        model,
        SelectOutput()
    )
    new_model.eval()"""

    # setup loss for attributes
    attr_criterion = []
    for j in range(n_attributes):
        attr_criterion.append(torch.nn.BCELoss())

    # setup dataset & dataloader
    data_dir = 'C:/Users/Christian Stoll/Documents/Studium/Master/aaa Thesis/MTConceptBottleneck/data/CUB_200_2011/CUB_processed/class_attr_data_10/train.pkl'
    loader = load_data([data_dir], use_attr=use_attr, no_img=False, uncertain_label=False, batch_size=1, image_dir='images',
                       n_class_attr=312)  # -> refer to dataset.py for difference, when changing 'use_attr'

    cnt = 0
    for data_idx, data in enumerate(loader):
        # data: [img_as_tensor, class_number_as_tensor, attr_labels_as_listOfTensors -> containing a single 0 or 1]
        inputs, labels, attr_labels = data
        attr_labels = torch.stack(attr_labels).t()
        adv_attr = attr_labels.clone()

        # invert num_attr randomly chosen labels
        for i in range(num_attr):
            pos = random.randint(0, n_attributes-1)
            if adv_attr[0, pos] == 1:
                adv_attr[0, pos] = 0
                #print('hi')
            else:
                adv_attr[0, pos] = 1

        inputs_adv = torch.autograd.Variable(inputs).cuda()
        labels_var = torch.autograd.Variable(labels).cuda() # not required here (for now)
        adv_attr_var = torch.autograd.Variable(adv_attr).float().cuda()

        class_outputs, attr_prediction_sigmoid = full_model(inputs_adv)

        # TODO include test if concept predictions are wrong/bad using this here
        """print(class_outputs)
        print(class_outputs.max(1, keepdim=True)[1])
        exit()"""
        # activate gradient tracking for input tensor -> cast to float to be able to optimize it
        inputs_adv.float()
        inputs_adv.requires_grad = True

        """atk = torchattacks.FGSM(model=full_model, eps=0.02)
        perturbed = atk(inputs_adv, labels_var)"""

        # initialize optimizer with learning rate and input_tensor
        optimizer = optim.LBFGS([inputs_adv], lr=0.0001, max_iter=1)

        # apply subsequent updates to input image to minimize loss, use LBFGS
        def closure():
            optimizer.zero_grad()

            # run forward pass (in eval mode)
            class_outputs, attr_prediction_sigmoid = full_model(inputs_adv)

            # apply binary cross_entropy to compute the loss of predicted concepts wrt. labels
            losses = []
            for i in range(n_attributes):
                # 2nd unsqueeze in target to model batch size (everything else is deprecated)
                losses.append(attr_criterion[i](attr_prediction_sigmoid[i].type(torch.cuda.FloatTensor),
                                                adv_attr_var[0, i].unsqueeze(0).unsqueeze(0)))
            total_loss = sum(losses) / n_attributes
            #total_loss.backward()

            loss = F.cross_entropy(class_outputs, labels_var)
            loss.backward()

            # substract loss from gradient
            inputs_adv.grad = inputs_adv.grad + eps * loss
            return loss

        optimizer.step(closure)

        # get values in range of 0,1 for an float type image
        perturbed = inputs_adv.clamp(0, 1)

        # get the newly predicted class
        adv_class_pred, adv_attr_pred_sigmoid = full_model(perturbed)
        adv_attr_pred_sigmoid = torch.cat(adv_attr_pred_sigmoid, dim=1)
        adv_class_pred = adv_class_pred.max(1, keepdim=True)[1]

        print('adversarial class: ', adv_class_pred)

        adv_attr_pred = adv_attr_pred_sigmoid.detach().cpu()
        label_diff = (attr_labels - adv_attr_pred).squeeze()
        print('label difference: ', label_diff)

        """plt.plot(label_diff.detach().cpu().numpy(), ls=':')
        plt.show()"""
        al = attr_labels.squeeze().numpy()
        ld = label_diff.numpy()
        aap = adv_attr_pred.squeeze().numpy()
        print(al)
        print(ld)
        print(aap)
        #pl = np.stack([al, ld, aap])
        pl = np.stack([aap, ld, al])
        print(pl.shape)
        print(pl)

        fig, axs = plt.subplots(1, 1)
        pcm = axs.pcolormesh(pl, cmap='gray')

        """pcm = axs[0].pcolormesh(al, cmap='gray')
        pcm = axs[1].pcolormesh(ld, cmap='gray')
        pcm = axs[2].pcolormesh(aap, cmap='gray')"""
        fig.colorbar(pcm, ax=axs)
        #fig.axes.get_yaxis().set_visible(False)
        #axs.set_yticklabels(['original', 'difference', 'perturbed'])
        plt.show()

        # get the faked image on the cpu
        perturbed_img = perturbed[0].detach().cpu()

        # show image, adv, difference etc.
        #title = 'manipulate image to change attribute predictions'
        title = str(cnt) + '_old_' + str(labels.item()) + '_new_' + str(adv_class_pred.item()) + '.png'
        show_adversarial_images(inverse_normalization(inputs[0]), inverse_normalization(perturbed_img), title=title)

        cnt += 1

        if cnt > 5:
            break
        #break